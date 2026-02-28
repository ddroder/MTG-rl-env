package rl;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import forge.LobbyPlayer;
import forge.deck.Deck;
import forge.deck.io.DeckSerializer;
import forge.game.Game;
import forge.game.GameRules;
import forge.game.GameType;
import forge.game.Match;
import forge.game.card.Card;
import forge.game.phase.PhaseHandler;
import forge.game.player.RegisteredPlayer;
import forge.ai.ComputerUtil;
import forge.ai.PlayerControllerAi;
import forge.gui.GuiBase;
import forge.gui.interfaces.IGuiBase;
import forge.item.PaperCard;
import forge.localinstance.properties.ForgeConstants;
import forge.localinstance.skin.FSkinProp;
import forge.localinstance.skin.ISkinImage;
import forge.model.FModel;
import forge.player.GamePlayerUtil;
import forge.gamemodes.match.HostedMatch;
import forge.gui.download.GuiDownloadService;
import forge.sound.IAudioClip;
import forge.sound.IAudioMusic;
import forge.util.FSerializableFunction;
import forge.util.ImageFetcher;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.function.Consumer;

import org.jupnp.UpnpServiceConfiguration;

import java.io.*;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

/**
 * ForgeEnvServer (rl_v0)
 *
 * NOTE: This file was reconstructed after an accidental overwrite.
 * It contains the current feature set: enumerated main-phase actions, action_id stepping,
 * richer snapshot state (hand names + mana proxies + commander info), and a robust,
 * non-blocking /step handler.
 */
public class ForgeEnvServer {

    // Keep consistent with snapshotState battlefield encoding.
    private static final int BF_N = 12;

    private static forge.game.player.Player opponentOf(Game g, forge.game.player.Player me) {
        if (g == null || me == null) return null;
        for (forge.game.player.Player p : g.getPlayers()) {
            if (p != null && p != me) return p;
        }
        return null;
    }

    private static List<Card> battlefieldSorted(forge.game.player.Player p) {
        if (p == null) return java.util.Collections.emptyList();
        List<Card> bf = new ArrayList<>(p.getCardsIn(forge.game.zone.ZoneType.Battlefield));
        bf.sort((a, b) -> {
            int c = a.getName().compareToIgnoreCase(b.getName());
            if (c != 0) return c;
            return Integer.compare(a.getId(), b.getId());
        });
        return bf;
    }

    private static Card battlefieldCardAtSlot(forge.game.player.Player p, int slot) {
        if (slot < 0 || slot >= BF_N) return null;
        List<Card> bf = battlefieldSorted(p);
        if (slot >= bf.size()) return null;
        return bf.get(slot);
    }

    private static forge.game.GameObject resolveSingleTargetSpec(String spec, forge.game.player.Player me, Game g) {
        if (spec == null || me == null || g == null) return null;
        String s = spec.trim();
        forge.game.player.Player opp = opponentOf(g, me);
        if (opp == null) return null;

        if (s.equalsIgnoreCase("TGT_P2")) {
            return opp;
        }
        if (s.toUpperCase().startsWith("TGT_P2_BF:")) {
            try {
                int slot = Integer.parseInt(s.substring("TGT_P2_BF:".length()).trim());
                return battlefieldCardAtSlot(opp, slot);
            } catch (Throwable ignored) {
                return null;
            }
        }
        if (s.toUpperCase().startsWith("TGT_P1_BF:")) {
            try {
                int slot = Integer.parseInt(s.substring("TGT_P1_BF:".length()).trim());
                return battlefieldCardAtSlot(me, slot);
            } catch (Throwable ignored) {
                return null;
            }
        }
        return null;
    }

    private static boolean isSingleTargetSA(forge.game.spellability.SpellAbility sa) {
        if (sa == null || !sa.usesTargeting()) return false;
        try {
            return sa.getMinTargets() == 1 && sa.getMaxTargets() == 1;
        } catch (Throwable ignored) {
            return false;
        }
    }

    private static volatile Game currentGame = null;
    private static volatile Match currentMatch = null;

    /** Per-game controller context so old game threads can't interfere with the current one. */
    private static final class GameContext {
        final BlockingQueue<String> q = new ArrayBlockingQueue<>(1);
        final java.util.concurrent.atomic.AtomicBoolean waiting = new java.util.concurrent.atomic.AtomicBoolean(false);
        volatile List<String> cachedLegalActions = null;
        volatile long cachedAtNanos = 0L;
    }

    private static volatile GameContext currentCtx = new GameContext();

    private static volatile boolean doneFlag = false;
    private static volatile String winnerName = null;

    // reward shaping state (last enqueued decision snapshot)
    private static volatile Integer lastP1Life = null;
    private static volatile Integer lastP2Life = null;
    private static volatile Integer lastP1Creatures = null;
    private static volatile Integer lastP2Creatures = null;

    private static String esc(String s) {
        if (s == null) return "";
        // Minimal JSON string escaping
        String out = s
                .replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
        // Strip other control characters
        out = out.replaceAll("[\\x00-\\x08\\x0B\\x0C\\x0E-\\x1F]", " ");
        return out;
    }

    private static String json(Map<String, Object> m) {
        StringBuilder sb = new StringBuilder();
        sb.append("{");
        boolean first = true;
        for (Map.Entry<String, Object> e : m.entrySet()) {
            if (!first) sb.append(",");
            first = false;
            sb.append("\"").append(esc(e.getKey())).append("\":");
            Object v = e.getValue();
            if (v == null) {
                sb.append("null");
            } else if (v instanceof Number || v instanceof Boolean) {
                sb.append(v.toString());
            } else if (v instanceof List) {
                @SuppressWarnings("unchecked")
                List<Object> list = (List<Object>) v;
                sb.append("[");
                boolean lf = true;
                for (Object o : list) {
                    if (!lf) sb.append(",");
                    lf = false;
                    sb.append("\"").append(esc(String.valueOf(o))).append("\"");
                }
                sb.append("]");
            } else {
                sb.append("\"").append(esc(String.valueOf(v))).append("\"");
            }
        }
        sb.append("}");
        return sb.toString();
    }

    private static void write(HttpExchange ex, int code, String body) throws IOException {
        byte[] b = body.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().add("content-type", "application/json; charset=utf-8");
        ex.sendResponseHeaders(code, b.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(b);
        }
    }

    private static String readBody(HttpExchange ex) throws IOException {
        try (InputStream is = ex.getRequestBody()) {
            return new String(is.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    private static Deck loadCommanderDeckByFilename(String filename) {
        File base = new File(ForgeConstants.DECK_COMMANDER_DIR);
        File f = new File(base, filename);
        if (!f.exists()) throw new RuntimeException("Deck not found: " + f.getAbsolutePath());
        return DeckSerializer.fromFile(f);
    }

    private static Map<String, Object> snapshotState() {
        Game g = currentGame;
        Integer turn = null;
        String phase = null;
        Integer phaseId = null;
        String priority = null;

        Integer p1Life = null;
        Integer p2Life = null;
        Integer p1Hand = null;
        Integer p2Hand = null;
        Integer p1Creatures = null;
        Integer p2Creatures = null;
        Integer p1Lands = null;
        Integer p2Lands = null;
        Integer p1UntappedLands = null;
        Integer p2UntappedLands = null;
        Integer p1ManaR = null, p1ManaG = null, p1ManaU = null, p1ManaB = null, p1ManaW = null;
        Integer p2ManaR = null, p2ManaG = null, p2ManaU = null, p2ManaB = null, p2ManaW = null;
        String p1CommanderZone = null;
        Integer p1CommanderCast = null;
        List<String> p1HandNames = null;

        // Commander debug flags
        Integer p1CmdInCommand = null;
        Integer p1CmdCanPlay = null;
        Integer p1CmdCanPay = null;
        Integer p1CmdCastable = null;
        Integer p1CmdSaCount = null;
        String p1CmdSa0 = null;
        String p1CmdSa0Desc = null;
        String p1Priority = null;
        Integer stackSize = null;
        String phaseName = null;

        // Battlefield fixed-size features
        List<String> p1BfNames = null, p2BfNames = null;
        List<Integer> p1BfTypes = null, p2BfTypes = null;
        List<Integer> p1BfTapped = null, p2BfTapped = null;
        List<Integer> p1BfSick = null, p2BfSick = null;
        List<Integer> p1BfPow = null, p2BfPow = null;
        List<Integer> p1BfTgh = null, p2BfTgh = null;
        List<Integer> p1BfIsCmd = null, p2BfIsCmd = null;

        if (g != null) {
            PhaseHandler ph = g.getPhaseHandler();
            if (ph != null) {
                turn = ph.getTurn();
                var phType = ph.getPhase();
                phase = phType != null ? phType.name() : null;
                phaseId = phType != null ? phType.ordinal() : null;
                priority = ph.getPriorityPlayer() != null ? ph.getPriorityPlayer().getName() : null;
            }
            phaseName = phase;
            p1Priority = priority;
            try {
                if (g.getStack() != null) stackSize = g.getStack().size();
            } catch (Throwable ignored) {}

            if (g.getPlayers().size() >= 2) {
                var p1 = g.getPlayers().get(0);
                var p2 = g.getPlayers().get(1);

                try {
                    p1Life = p1.getLife();
                    p2Life = p2.getLife();

                    var p1HandCards = p1.getCardsIn(forge.game.zone.ZoneType.Hand);
                    var p2HandCards = p2.getCardsIn(forge.game.zone.ZoneType.Hand);
                    p1Hand = p1HandCards.size();
                    p2Hand = p2HandCards.size();

                    p1Creatures = p1.getCreaturesInPlay().size();
                    p2Creatures = p2.getCreaturesInPlay().size();

                    var p1Battle = p1.getCardsIn(forge.game.zone.ZoneType.Battlefield);
                    var p2Battle = p2.getCardsIn(forge.game.zone.ZoneType.Battlefield);

                    // Battlefield snapshot (fixed size, deterministic ordering)
                    try {
                        final int BF_N = 12;
                        List<Card> p1Bf = new ArrayList<>(p1Battle);
                        List<Card> p2Bf = new ArrayList<>(p2Battle);
                        p1Bf.sort((a, b) -> {
                            int c = a.getName().compareToIgnoreCase(b.getName());
                            if (c != 0) return c;
                            return Integer.compare(a.getId(), b.getId());
                        });
                        p2Bf.sort((a, b) -> {
                            int c = a.getName().compareToIgnoreCase(b.getName());
                            if (c != 0) return c;
                            return Integer.compare(a.getId(), b.getId());
                        });

                        p1BfNames = new ArrayList<>(BF_N);
                        p1BfTypes = new ArrayList<>(BF_N);
                        p1BfTapped = new ArrayList<>(BF_N);
                        p1BfSick = new ArrayList<>(BF_N);
                        p1BfPow = new ArrayList<>(BF_N);
                        p1BfTgh = new ArrayList<>(BF_N);
                        p1BfIsCmd = new ArrayList<>(BF_N);

                        p2BfNames = new ArrayList<>(BF_N);
                        p2BfTypes = new ArrayList<>(BF_N);
                        p2BfTapped = new ArrayList<>(BF_N);
                        p2BfSick = new ArrayList<>(BF_N);
                        p2BfPow = new ArrayList<>(BF_N);
                        p2BfTgh = new ArrayList<>(BF_N);
                        p2BfIsCmd = new ArrayList<>(BF_N);

                        java.util.function.Function<Card, Integer> typeMask = (Card c) -> {
                            int m = 0;
                            if (c.isCreature()) m |= 1;
                            if (c.isLand()) m |= 2;
                            if (c.isArtifact()) m |= 4;
                            if (c.isEnchantment()) m |= 8;
                            if (c.isPlaneswalker()) m |= 16;
                            return m;
                        };

                        java.util.function.Function<Card, Integer> powFn = (Card c) -> {
                            if (!c.isCreature()) return 0;
                            int pwr = c.getNetPower();
                            return Math.max(-20, Math.min(20, pwr));
                        };
                        java.util.function.Function<Card, Integer> tghFn = (Card c) -> {
                            if (!c.isCreature()) return 0;
                            int t = c.getNetToughness();
                            return Math.max(-20, Math.min(20, t));
                        };

                        java.util.Set<Integer> p1CmdIds = new java.util.HashSet<>();
                        for (Card cc : p1.getCommanders()) p1CmdIds.add(cc.getId());
                        java.util.Set<Integer> p2CmdIds = new java.util.HashSet<>();
                        for (Card cc : p2.getCommanders()) p2CmdIds.add(cc.getId());

                        for (int bi = 0; bi < BF_N; bi++) {
                            Card c = bi < p1Bf.size() ? p1Bf.get(bi) : null;
                            if (c == null) {
                                p1BfNames.add("");
                                p1BfTypes.add(0);
                                p1BfTapped.add(0);
                                p1BfSick.add(0);
                                p1BfPow.add(0);
                                p1BfTgh.add(0);
                                p1BfIsCmd.add(0);
                            } else {
                                p1BfNames.add(c.getName());
                                p1BfTypes.add(typeMask.apply(c));
                                p1BfTapped.add(c.isTapped() ? 1 : 0);
                                p1BfSick.add(c.isSick() ? 1 : 0);
                                p1BfPow.add(powFn.apply(c));
                                p1BfTgh.add(tghFn.apply(c));
                                p1BfIsCmd.add(p1CmdIds.contains(c.getId()) ? 1 : 0);
                            }
                        }
                        for (int bi = 0; bi < BF_N; bi++) {
                            Card c = bi < p2Bf.size() ? p2Bf.get(bi) : null;
                            if (c == null) {
                                p2BfNames.add("");
                                p2BfTypes.add(0);
                                p2BfTapped.add(0);
                                p2BfSick.add(0);
                                p2BfPow.add(0);
                                p2BfTgh.add(0);
                                p2BfIsCmd.add(0);
                            } else {
                                p2BfNames.add(c.getName());
                                p2BfTypes.add(typeMask.apply(c));
                                p2BfTapped.add(c.isTapped() ? 1 : 0);
                                p2BfSick.add(c.isSick() ? 1 : 0);
                                p2BfPow.add(powFn.apply(c));
                                p2BfTgh.add(tghFn.apply(c));
                                p2BfIsCmd.add(p2CmdIds.contains(c.getId()) ? 1 : 0);
                            }
                        }
                    } catch (Throwable t) {
                        // Avoid breaking the whole snapshot if battlefield encoding fails.
                        t.printStackTrace();
                    }
                    p1Lands = (int) p1Battle.stream().filter(Card::isLand).count();
                    p2Lands = (int) p2Battle.stream().filter(Card::isLand).count();

                    int u1 = 0, u2 = 0;
                    int r1 = 0, g1 = 0, u1c = 0, b1 = 0, w1 = 0;
                    int r2 = 0, g2 = 0, u2c = 0, b2 = 0, w2 = 0;
                    for (Card c : p1Battle) {
                        if (!c.isLand() || c.isTapped()) continue;
                        u1++;
                        if (c.getType().hasSubtype("Mountain")) r1++;
                        if (c.getType().hasSubtype("Forest")) g1++;
                        if (c.getType().hasSubtype("Island")) u1c++;
                        if (c.getType().hasSubtype("Swamp")) b1++;
                        if (c.getType().hasSubtype("Plains")) w1++;
                    }
                    for (Card c : p2Battle) {
                        if (!c.isLand() || c.isTapped()) continue;
                        u2++;
                        if (c.getType().hasSubtype("Mountain")) r2++;
                        if (c.getType().hasSubtype("Forest")) g2++;
                        if (c.getType().hasSubtype("Island")) u2c++;
                        if (c.getType().hasSubtype("Swamp")) b2++;
                        if (c.getType().hasSubtype("Plains")) w2++;
                    }
                    p1UntappedLands = u1;
                    p2UntappedLands = u2;
                    p1ManaR = r1; p1ManaG = g1; p1ManaU = u1c; p1ManaB = b1; p1ManaW = w1;
                    p2ManaR = r2; p2ManaG = g2; p2ManaU = u2c; p2ManaB = b2; p2ManaW = w2;

                    if (!p1.getCommanders().isEmpty()) {
                        Card cmd = p1.getCommanders().get(0);
                        var z = g.getZoneOf(cmd);
                        p1CommanderZone = z != null ? z.getZoneType().name() : null;
                        p1CommanderCast = p1.getCommanderCast(cmd);

                        boolean inCmd = (z != null && z.getZoneType() != null && "Command".equalsIgnoreCase(z.getZoneType().name()));
                        p1CmdInCommand = inCmd ? 1 : 0;

                        // Determine castability (best-effort)
                        boolean canPlay = false;
                        boolean canPay = false;
                        try {
                            forge.util.collect.FCollectionView<forge.game.spellability.SpellAbility> sas = cmd.getSpellAbilities();
                            p1CmdSaCount = sas != null ? sas.size() : 0;
                            if (sas != null) {
                                for (forge.game.spellability.SpellAbility sa : sas) {
                                    if (sa == null) continue;
                                    if (p1CmdSa0 == null) {
                                        p1CmdSa0 = String.valueOf(sa);
                                        try { p1CmdSa0Desc = sa.getDescription(); } catch (Throwable ignored) {}
                                    }
                                    if (!sa.isSpell()) continue;
                                    sa.setActivatingPlayer(p1);
                                    // canPlay(true) is used by Forge for "ignore timing" (and may also bypass zone restrictions)
                                    canPlay = sa.canPlay(true);
                                    canPay = forge.ai.ComputerUtilMana.canPayManaCost(sa, p1, 0, true);
                                    break;
                                }
                            }
                        } catch (Throwable t) {
                            t.printStackTrace();
                        }
                        p1CmdCanPlay = canPlay ? 1 : 0;
                        p1CmdCanPay = canPay ? 1 : 0;
                        p1CmdCastable = (inCmd && canPlay && canPay) ? 1 : 0;
                    }

                    final int HAND_N = 10;
                    p1HandNames = new ArrayList<>(HAND_N);
                    int i = 0;
                    for (Card c : p1HandCards) {
                        if (i >= HAND_N) break;
                        p1HandNames.add(c.getName());
                        i++;
                    }
                    while (p1HandNames.size() < HAND_N) p1HandNames.add("");

                } catch (Throwable ignored) {
                }
            }
        }

        Map<String, Object> m = new LinkedHashMap<>();
        m.put("turn", turn);
        m.put("phase", phase);
        m.put("phase_id", phaseId);
        m.put("priority_player", priority);

        m.put("p1_life", p1Life);
        m.put("p2_life", p2Life);
        m.put("p1_hand", p1Hand);
        m.put("p2_hand", p2Hand);
        m.put("p1_creatures", p1Creatures);
        m.put("p2_creatures", p2Creatures);
        m.put("p1_lands", p1Lands);
        m.put("p2_lands", p2Lands);
        m.put("p1_untapped_lands", p1UntappedLands);
        m.put("p2_untapped_lands", p2UntappedLands);
        m.put("p1_mana_r", p1ManaR);
        m.put("p1_mana_g", p1ManaG);
        m.put("p1_mana_u", p1ManaU);
        m.put("p1_mana_b", p1ManaB);
        m.put("p1_mana_w", p1ManaW);
        m.put("p2_mana_r", p2ManaR);
        m.put("p2_mana_g", p2ManaG);
        m.put("p2_mana_u", p2ManaU);
        m.put("p2_mana_b", p2ManaB);
        m.put("p2_mana_w", p2ManaW);
        m.put("p1_commander_zone", p1CommanderZone);
        m.put("p1_commander_cast", p1CommanderCast);
        m.put("p1_hand_names", p1HandNames);

        // Commander debug flags
        m.put("p1_cmd_in_command", p1CmdInCommand);
        m.put("p1_cmd_can_play", p1CmdCanPlay);
        m.put("p1_cmd_can_pay", p1CmdCanPay);
        m.put("p1_cmd_castable", p1CmdCastable);
        m.put("p1_cmd_sa_count", p1CmdSaCount);
        m.put("p1_cmd_sa0", p1CmdSa0);
        m.put("p1_cmd_sa0_desc", p1CmdSa0Desc);
        m.put("phase_name", phaseName);
        m.put("priority_player", p1Priority);
        m.put("stack_size", stackSize);

        // Battlefield fixed-size lists
        if (p1BfNames != null) m.put("p1_bf_names", p1BfNames);
        if (p1BfTypes != null) m.put("p1_bf_types", p1BfTypes);
        if (p1BfTapped != null) m.put("p1_bf_tapped", p1BfTapped);
        if (p1BfSick != null) m.put("p1_bf_sick", p1BfSick);
        if (p1BfPow != null) m.put("p1_bf_pow", p1BfPow);
        if (p1BfTgh != null) m.put("p1_bf_tgh", p1BfTgh);
        if (p1BfIsCmd != null) m.put("p1_bf_is_cmd", p1BfIsCmd);

        if (p2BfNames != null) m.put("p2_bf_names", p2BfNames);
        if (p2BfTypes != null) m.put("p2_bf_types", p2BfTypes);
        if (p2BfTapped != null) m.put("p2_bf_tapped", p2BfTapped);
        if (p2BfSick != null) m.put("p2_bf_sick", p2BfSick);
        if (p2BfPow != null) m.put("p2_bf_pow", p2BfPow);
        if (p2BfTgh != null) m.put("p2_bf_tgh", p2BfTgh);
        if (p2BfIsCmd != null) m.put("p2_bf_is_cmd", p2BfIsCmd);

        m.put("done", doneFlag);
        m.put("winner", winnerName);
        GameContext ctx = currentCtx;
        m.put("external_waiting", ctx != null && ctx.waiting.get());

        return m;
    }

    private static List<String> legalActionsSnapshotUncached() {
        List<String> acts = new ArrayList<>();
        acts.add("PASS");

        Game g = currentGame;
        if (g == null || g.getPlayers().isEmpty()) return acts;

        PhaseHandler ph = g.getPhaseHandler();
        if (ph == null) return acts;

        String phase = ph.getPhase() != null ? ph.getPhase().name() : null;
        if (phase == null) return acts;

        // Main phases: enumerate playable lands/spells/activations in deterministic order
        if ("MAIN1".equals(phase) || "MAIN2".equals(phase)) {
            forge.game.player.Player p = g.getPlayers().get(0);
            List<Card> hand = new ArrayList<>(p.getCardsIn(forge.game.zone.ZoneType.Hand));

            // Commander cast (first commander only) — keep near the top of the action list
            // NOTE: Forge's normal spell ability for a commander returns canPlay()==false from the Command zone.
            // There is also canPlay(true) (ignore timing). We'll use that for commander-cast permission rather than moving zones.
            try {
                if (!p.getCommanders().isEmpty()) {
                    Card cmd = p.getCommanders().get(0);
                    var z = g.getZoneOf(cmd);
                    boolean inCmd = (z != null && z.getZoneType() != null && "Command".equalsIgnoreCase(z.getZoneType().name()));
                    boolean stackEmpty = true;
                    try { stackEmpty = (g.getStack() == null || g.getStack().size() == 0); } catch (Throwable ignored) {}
                    boolean ourPriority = false;
                    try { ourPriority = (ph.getPriorityPlayer() != null && "External-1".equals(ph.getPriorityPlayer().getName())); } catch (Throwable ignored) {}
                    boolean canPay = false;
                    try {
                        for (forge.game.spellability.SpellAbility sa : cmd.getSpellAbilities()) {
                            if (sa == null || !sa.isSpell()) continue;
                            sa.setActivatingPlayer(p);
                            if (forge.ai.ComputerUtilMana.canPayManaCost(sa, p, 0, true)) { canPay = true; break; }
                        }
                    } catch (Throwable ignored) {}
                    // NOTE: Forge's canPlay() (even canPlay(true)) does not seem to allow casting from Command zone.
                    // We expose CAST_COMMANDER based on payability, and the controller handles the cast.
                    if (inCmd && stackEmpty && ourPriority && canPay) {
                        acts.add("CAST_COMMANDER");
                    }
                }
            } catch (Throwable ignored) {}

            ArrayList<String> lands = new ArrayList<>();
            HashSet<String> seenLands = new HashSet<>();
            for (Card c : hand) {
                if (!c.isLand()) continue;
                if (!seenLands.add(c.getName())) continue;
                boolean ok = false;
                for (forge.game.spellability.SpellAbility sa : c.getSpellAbilities()) {
                    if (sa == null || !sa.isLandAbility()) continue;
                    sa.setActivatingPlayer(p);
                    try {
                        if (p.canPlayLand(c, false, sa) && sa.canPlay()) { ok = true; break; }
                    } catch (Throwable ignored) {}
                }
                if (ok) lands.add("PLAY_LAND:" + c.getName());
            }
            lands.sort(String.CASE_INSENSITIVE_ORDER);
            acts.addAll(lands);

            // Spells: support non-targeting casts, and minimal single-target casts with explicit targets.
            // Target encoding:
            //   CAST:<cardName>:TGT_P2
            //   CAST:<cardName>:TGT_P2_BF:<slot>
            ArrayList<String> spells = new ArrayList<>();
            HashSet<String> seenSpells = new HashSet<>();
            ArrayList<String> spellNames = new ArrayList<>();
            for (Card c : hand) {
                if (c == null || c.isLand()) continue;
                if (seenSpells.add(c.getName())) spellNames.add(c.getName());
            }
            spellNames.sort(String.CASE_INSENSITIVE_ORDER);

            forge.game.player.Player opp = opponentOf(g, p);
            for (int ni = 0; ni < spellNames.size() && ni < 12; ni++) {
                String nm = spellNames.get(ni);
                Card sample = null;
                for (Card c : hand) {
                    if (c != null && !c.isLand() && c.getName().equalsIgnoreCase(nm)) { sample = c; break; }
                }
                if (sample == null) continue;

                forge.game.spellability.SpellAbility chosen = null;
                for (forge.game.spellability.SpellAbility sa : sample.getSpellAbilities()) {
                    if (sa == null || !sa.isSpell()) continue;
                    sa.setActivatingPlayer(p);
                    try {
                        if (!sa.canPlay()) continue;
                        if (!forge.ai.ComputerUtilMana.canPayManaCost(sa, p, 0, true)) continue;
                    } catch (Throwable ignored) { continue; }
                    chosen = sa;
                    chosen.setTargetingPlayer(p);
                    break;
                }
                if (chosen == null) continue;

                if (!chosen.usesTargeting()) {
                    spells.add("CAST:" + nm);
                    continue;
                }

                // Minimal target support: exactly one target, and only opponent player or battlefield slots.
                if (!isSingleTargetSA(chosen) || opp == null) continue;

                try {
                    if (chosen.canTarget(opp)) spells.add("CAST:" + nm + ":TGT_P2");
                } catch (Throwable ignored) {}
                for (int slot = 0; slot < BF_N; slot++) {
                    Card tgt = battlefieldCardAtSlot(opp, slot);
                    if (tgt == null) continue;
                    try {
                        if (chosen.canTarget(tgt)) spells.add("CAST:" + nm + ":TGT_P2_BF:" + slot);
                    } catch (Throwable ignored) {}
                }
                for (int slot = 0; slot < BF_N; slot++) {
                    Card tgt = battlefieldCardAtSlot(p, slot);
                    if (tgt == null) continue;
                    try {
                        if (chosen.canTarget(tgt)) spells.add("CAST:" + nm + ":TGT_P1_BF:" + slot);
                    } catch (Throwable ignored) {}
                }
            }
            spells.sort(String.CASE_INSENSITIVE_ORDER);
            acts.addAll(spells);

            // Activated abilities: support non-targeting activations and minimal single-target activations.
            // Target encoding:
            //   ACTIVATE:<permName>:<idx>:TGT_P2
            //   ACTIVATE:<permName>:<idx>:TGT_P2_BF:<slot>
            try {
                List<Card> bf = new ArrayList<>(p.getCardsIn(forge.game.zone.ZoneType.Battlefield));
                ArrayList<String> activates = new ArrayList<>();
                int cap = 40;
                forge.game.player.Player opp2 = opponentOf(g, p);

                for (Card perm : bf) {
                    if (perm == null) continue;
                    int idx = 0;
                    for (forge.game.spellability.SpellAbility sa : perm.getSpellAbilities()) {
                        if (sa == null) continue;
                        if (!sa.isActivatedAbility()) continue;
                        if (sa.isSpell() || sa.isLandAbility()) continue;
                        sa.setActivatingPlayer(p);
                        sa.setTargetingPlayer(p);

                        if (!sa.canPlay()) { idx++; continue; }
                        // Don't expose mana abilities (especially lands) as explicit RL actions;
                        // casting spells already triggers Forge's cost payment.
                        if (sa.isManaAbility()) { idx++; continue; }

                        if (!sa.usesTargeting()) {
                            activates.add("ACTIVATE:" + perm.getName() + ":" + idx);
                            idx++;
                            if (activates.size() >= cap) break;
                            continue;
                        }

                        // Minimal target support: exactly one target, and only opponent player or battlefield slots.
                        if (!isSingleTargetSA(sa) || opp2 == null) { idx++; continue; }

                        try {
                            if (sa.canTarget(opp2)) {
                                activates.add("ACTIVATE:" + perm.getName() + ":" + idx + ":TGT_P2");
                                if (activates.size() >= cap) { idx++; break; }
                            }
                        } catch (Throwable ignored) {}
                        for (int slot = 0; slot < BF_N && activates.size() < cap; slot++) {
                            Card tgt = battlefieldCardAtSlot(opp2, slot);
                            if (tgt == null) continue;
                            try {
                                if (sa.canTarget(tgt)) {
                                    activates.add("ACTIVATE:" + perm.getName() + ":" + idx + ":TGT_P2_BF:" + slot);
                                }
                            } catch (Throwable ignored) {}
                        }
                        for (int slot = 0; slot < BF_N && activates.size() < cap; slot++) {
                            Card tgt = battlefieldCardAtSlot(p, slot);
                            if (tgt == null) continue;
                            try {
                                if (sa.canTarget(tgt)) {
                                    activates.add("ACTIVATE:" + perm.getName() + ":" + idx + ":TGT_P1_BF:" + slot);
                                }
                            } catch (Throwable ignored) {}
                        }

                        idx++;
                        if (activates.size() >= cap) break;
                    }
                    if (activates.size() >= cap) break;
                }
                activates.sort(String.CASE_INSENSITIVE_ORDER);
                acts.addAll(activates);
            } catch (Throwable ignored) {}
        }

        // Combat: macro actions (incremental control)
        if ("COMBAT_DECLARE_ATTACKERS".equals(phase)) {
            acts.add("ATTACK_NONE");
            acts.add("ATTACK_ALL");
            acts.add("ATTACK_NON_SICK");
            acts.add("ATTACK_TOP_POWER_1");
            acts.add("ATTACK_TOP_POWER_2");
            acts.add("ATTACK_TOP_POWER_3");
            acts.add("USE_AI_COMBAT");
        }
        if ("COMBAT_DECLARE_BLOCKERS".equals(phase)) {
            // Blocking macros (coarse control): keep action space small but give the policy a lever.
            acts.add("BLOCK_NONE");
            acts.add("BLOCK_MAX");
            acts.add("BLOCK_TRADE");
            acts.add("BLOCK_CHUMP_IF_LETHAL");
            acts.add("USE_AI_COMBAT");
        }

        return acts;
    }

    private static List<String> legalActionsSnapshot() {
        GameContext ctx = currentCtx;
        if (ctx != null && ctx.waiting.get()) {
            List<String> c = ctx.cachedLegalActions;
            if (c != null && !c.isEmpty()) return c;
        }
        return legalActionsSnapshotUncached();
    }

    private static final class ExternalController extends PlayerControllerAi {
        private final GameContext ctx;

        ExternalController(Game game, forge.game.player.Player player, LobbyPlayer lp, GameContext ctx) {
            super(game, player, lp);
            this.ctx = ctx;
        }

        @Override
        public List<forge.game.spellability.SpellAbility> chooseSpellAbilityToPlay() {
            try {
                ctx.waiting.set(true);
                // Cache legal actions for this decision point (avoid recomputing on every HTTP call).
                try {
                    ctx.cachedLegalActions = legalActionsSnapshotUncached();
                    ctx.cachedAtNanos = System.nanoTime();
                } catch (Throwable ignored) {}

                // Never block forever: if the HTTP side stalls, default to PASS so the game can keep advancing.
                String a = ctx.q.poll(5, java.util.concurrent.TimeUnit.SECONDS);
                if (a == null) return null;
                String act = a.trim();
                if (act.equalsIgnoreCase("PASS")) return null;

                forge.game.player.Player p = getPlayer();
                List<Card> hand = new ArrayList<>(p.getCardsIn(forge.game.zone.ZoneType.Hand));

                if (act.toUpperCase().startsWith("PLAY_LAND:")) {
                    String want = act.substring("PLAY_LAND:".length()).trim();
                    for (Card c : hand) {
                        if (!c.isLand()) continue;
                        if (!c.getName().equalsIgnoreCase(want)) continue;
                        for (forge.game.spellability.SpellAbility sa : c.getSpellAbilities()) {
                            if (sa != null && sa.isLandAbility()) {
                                sa.setActivatingPlayer(p);
                                if (sa.canPlay()) return List.of(sa);
                            }
                        }
                    }
                    return null;
                }

                if (act.equalsIgnoreCase("CAST_COMMANDER")) {
                    try {
                        if (!p.getCommanders().isEmpty()) {
                            Card cmd = p.getCommanders().get(0);
                            Game game = getGame();
                            var z = game != null ? game.getZoneOf(cmd) : null;
                            boolean inCmd = (z != null && z.getZoneType() != null && "Command".equalsIgnoreCase(z.getZoneType().name()));

                            // Forge doesn't allow canPlay() from Command zone for commander.
                            // Pragmatic hack: move to hand, then cast normally.
                            boolean casted = false;
                            try {
                                if (inCmd && game != null) {
                                    cmd = game.getAction().moveToHand(cmd, null);
                                }
                            } catch (Throwable ignored) {}

                            for (forge.game.spellability.SpellAbility sa : cmd.getSpellAbilities()) {
                                if (sa == null || !sa.isSpell()) continue;
                                sa.setActivatingPlayer(p);
                                if (!sa.canPlay()) continue;
                                if (!forge.ai.ComputerUtilMana.canPayManaCost(sa, p, 0, true)) continue;
                                ComputerUtil.playStack(sa, p, game);
                                casted = true;
                                break;
                            }

                            if (!casted) {
                                // If something went wrong, put it back to Command.
                                try { game.getAction().moveToCommand(cmd, null); } catch (Throwable ignored) {}
                            } else {
                                // track commander cast count (tax correctness is imperfect with move-to-hand)
                                try { p.incCommanderCast(cmd); } catch (Throwable ignored) {}
                            }
                        }
                    } catch (Throwable ignored) {}
                    return null;
                }

                if (act.toUpperCase().startsWith("CAST:")) {
                    // CAST:<cardName> (no targets)
                    // CAST:<cardName>:TGT_P2
                    // CAST:<cardName>:TGT_P2_BF:<slot>
                    String rest = act.substring("CAST:".length()).trim();
                    String want = rest;
                    String tgtSpec = null;
                    int tpos = rest.indexOf(":TGT_");
                    if (tpos >= 0) {
                        want = rest.substring(0, tpos).trim();
                        tgtSpec = rest.substring(tpos + 1).trim();
                    }

                    for (Card c : hand) {
                        if (c == null || c.isLand()) continue;
                        if (!c.getName().equalsIgnoreCase(want)) continue;
                        for (forge.game.spellability.SpellAbility sa : c.getSpellAbilities()) {
                            if (sa == null || !sa.isSpell()) continue;
                            sa.setActivatingPlayer(p);
                            sa.setTargetingPlayer(p);
                            if (!sa.canPlay()) continue;

                            if (tgtSpec != null) {
                                if (!sa.usesTargeting() || !isSingleTargetSA(sa)) continue;
                                forge.game.GameObject tgt = resolveSingleTargetSpec(tgtSpec, p, getGame());
                                if (tgt == null) continue;
                                sa.resetTargets();
                                if (!sa.canTarget(tgt)) continue;
                                sa.getTargets().add(tgt);
                                if (!sa.isTargetNumberValid()) continue;
                            } else {
                                // We currently do not support automatic target selection.
                                if (sa.usesTargeting()) continue;
                            }

                            ComputerUtil.playStack(sa, p, getGame());
                            return null;
                        }
                    }
                    return null;
                }

                if (act.toUpperCase().startsWith("ACTIVATE:")) {
                    try {
                        // ACTIVATE:<name>:<idx>
                        // ACTIVATE:<name>:<idx>:TGT_P2
                        // ACTIVATE:<name>:<idx>:TGT_P2_BF:<slot>
                        String rest = act.substring("ACTIVATE:".length()).trim();
                        String left = rest;
                        String tgtSpec = null;
                        int tpos = rest.indexOf(":TGT_");
                        if (tpos >= 0) {
                            left = rest.substring(0, tpos).trim();
                            tgtSpec = rest.substring(tpos + 1).trim();
                        }

                        int colon = left.lastIndexOf(':');
                        if (colon > 0) {
                            String want = left.substring(0, colon).trim();
                            int wantIdx = Integer.parseInt(left.substring(colon + 1).trim());

                            List<Card> bf = new ArrayList<>(p.getCardsIn(forge.game.zone.ZoneType.Battlefield));
                            for (Card perm : bf) {
                                if (perm == null) continue;
                                if (!perm.getName().equalsIgnoreCase(want)) continue;
                                int idx = 0;
                                for (forge.game.spellability.SpellAbility sa : perm.getSpellAbilities()) {
                                    if (sa == null) continue;
                                    if (!sa.isActivatedAbility()) continue;
                                    if (sa.isSpell() || sa.isLandAbility()) continue;
                                    if (idx != wantIdx) { idx++; continue; }

                                    sa.setActivatingPlayer(p);
                                    sa.setTargetingPlayer(p);
                                    if (!sa.canPlay()) return null;

                                    if (tgtSpec != null) {
                                        if (!sa.usesTargeting() || !isSingleTargetSA(sa)) return null;
                                        forge.game.GameObject tgt = resolveSingleTargetSpec(tgtSpec, p, getGame());
                                        if (tgt == null) return null;
                                        sa.resetTargets();
                                        if (!sa.canTarget(tgt)) return null;
                                        sa.getTargets().add(tgt);
                                        if (!sa.isTargetNumberValid()) return null;
                                    } else {
                                        if (sa.usesTargeting()) return null;
                                    }

                                    if (sa.isManaAbility()) {
                                        ComputerUtil.playNoStack(p, sa, getGame(), true);
                                    } else {
                                        ComputerUtil.playStack(sa, p, getGame());
                                    }
                                    return null;
                                }
                            }
                        }
                    } catch (Throwable ignored) {}
                    return null;
                }

                // fallback
                return null;
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return null;
            } finally {
                ctx.waiting.set(false);
                ctx.cachedLegalActions = null;
            }
        }

        @Override
        public void declareAttackers(forge.game.player.Player attacker, forge.game.combat.Combat combat) {
            try {
                ctx.waiting.set(true);
                try {
                    ctx.cachedLegalActions = legalActionsSnapshotUncached();
                    ctx.cachedAtNanos = System.nanoTime();
                } catch (Throwable ignored) {}
                // Never block forever; default to do-nothing if the HTTP side stalls.
                String act = ctx.q.poll(5, java.util.concurrent.TimeUnit.SECONDS);
                if (act == null) return;
                if (act.equalsIgnoreCase("USE_AI_COMBAT")) {
                    super.declareAttackers(attacker, combat);
                    return;
                }
                forge.game.player.Player defender = combat.getDefendingPlayers().isEmpty() ? null : combat.getDefendingPlayers().get(0);
                if (defender == null) return;

                java.util.List<Card> attackers = new java.util.ArrayList<>(forge.game.combat.CombatUtil.getPossibleAttackers(attacker));
                java.util.function.Predicate<Card> canAtk = c -> forge.game.combat.CombatUtil.canAttack(c, defender);

                if (act.equalsIgnoreCase("ATTACK_ALL")) {
                    for (Card c : attackers) {
                        if (canAtk.test(c)) combat.addAttacker(c, defender);
                    }
                    return;
                }

                if (act.equalsIgnoreCase("ATTACK_NON_SICK")) {
                    for (Card c : attackers) {
                        if (c != null && !c.isSick() && canAtk.test(c)) combat.addAttacker(c, defender);
                    }
                    return;
                }

                if (act.toUpperCase().startsWith("ATTACK_TOP_POWER_")) {
                    int n = 1;
                    try {
                        n = Integer.parseInt(act.substring("ATTACK_TOP_POWER_".length()).trim());
                    } catch (Throwable ignored) {}
                    final int want = Math.max(0, Math.min(n, 8));

                    attackers.removeIf(c -> c == null || !canAtk.test(c));
                    attackers.sort((a, b) -> {
                        int pa = a.getNetPower();
                        int pb = b.getNetPower();
                        if (pa != pb) return Integer.compare(pb, pa);
                        int ta = a.getNetToughness();
                        int tb = b.getNetToughness();
                        if (ta != tb) return Integer.compare(tb, ta);
                        return a.getName().compareToIgnoreCase(b.getName());
                    });

                    for (int i = 0; i < attackers.size() && i < want; i++) {
                        combat.addAttacker(attackers.get(i), defender);
                    }
                    return;
                }

                // default: do nothing
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            } finally {
                ctx.waiting.set(false);
                ctx.cachedLegalActions = null;
            }
        }

        @Override
        public void declareBlockers(forge.game.player.Player defender, forge.game.combat.Combat combat) {
            try {
                ctx.waiting.set(true);
                try {
                    ctx.cachedLegalActions = legalActionsSnapshotUncached();
                    ctx.cachedAtNanos = System.nanoTime();
                } catch (Throwable ignored) {}
                // Never block forever; default to do-nothing if the HTTP side stalls.
                String act = ctx.q.poll(5, java.util.concurrent.TimeUnit.SECONDS);
                if (act == null) return;
                if (act.equalsIgnoreCase("USE_AI_COMBAT")) {
                    super.declareBlockers(defender, combat);
                    return;
                }

                if (combat == null || defender == null) return;

                // Collect attackers and blockers
                java.util.List<Card> attackers = new java.util.ArrayList<>(combat.getAttackers());
                java.util.List<Card> blockers = new java.util.ArrayList<>(defender.getCreaturesInPlay());
                blockers.removeIf(b -> b == null || !forge.game.combat.CombatUtil.canBlock(b, combat));

                if (act.equalsIgnoreCase("BLOCK_NONE")) {
                    return;
                }

                // If we only chump when lethal, check lethal first
                if (act.equalsIgnoreCase("BLOCK_CHUMP_IF_LETHAL")) {
                    int life = 0;
                    try { life = defender.getLife(); } catch (Throwable ignored) {}
                    int incoming = 0;
                    for (Card a : attackers) {
                        if (a == null) continue;
                        incoming += Math.max(0, a.getNetPower());
                    }
                    if (incoming < life) {
                        return; // no blocks unless lethal
                    }
                    // else fall through to BLOCK_MAX behavior
                    act = "BLOCK_MAX";
                }

                // Sort attackers by power desc (try to reduce big hits first)
                attackers.removeIf(a -> a == null);
                attackers.sort((a, b) -> Integer.compare(b.getNetPower(), a.getNetPower()));

                // Sort blockers depending on macro
                if (act.equalsIgnoreCase("BLOCK_MAX")) {
                    // chump with lowest toughness first
                    blockers.sort((a, b) -> Integer.compare(a.getNetToughness(), b.getNetToughness()));
                } else if (act.equalsIgnoreCase("BLOCK_TRADE")) {
                    // try to use higher power blockers first to potentially kill attackers
                    blockers.sort((a, b) -> Integer.compare(b.getNetPower(), a.getNetPower()));
                }

                // Greedy assignment: one blocker per attacker (keeps it simple)
                java.util.Set<Card> used = new java.util.HashSet<>();
                for (Card attacker : attackers) {
                    Card best = null;
                    for (Card blocker : blockers) {
                        if (blocker == null || used.contains(blocker)) continue;
                        try {
                            if (!forge.game.combat.CombatUtil.canBlock(attacker, blocker, combat)) continue;
                        } catch (Throwable ignored) { continue; }

                        if (act.equalsIgnoreCase("BLOCK_TRADE")) {
                            // prefer blocks that can kill the attacker
                            boolean kills = false;
                            try { kills = blocker.getNetPower() >= attacker.getNetToughness(); } catch (Throwable ignored) {}
                            if (kills) { best = blocker; break; }
                            // otherwise consider as fallback
                            if (best == null) best = blocker;
                        } else {
                            // BLOCK_MAX: any legal blocker is fine, prefer smallest toughness (already sorted)
                            best = blocker;
                            break;
                        }
                    }
                    if (best != null) {
                        used.add(best);
                        try { combat.addBlocker(attacker, best); } catch (Throwable ignored) {}
                    }
                }

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            } finally {
                ctx.waiting.set(false);
                ctx.cachedLegalActions = null;
            }
        }
    }

    private static void startGameThread(String deckA, String deckB) {
        doneFlag = false;
        winnerName = null;
        if (currentCtx != null) currentCtx.waiting.set(false);

        new Thread(() -> {
            try {
                FModel.initialize(null, null);

                GameRules rules = new GameRules(GameType.Commander);
                rules.setAppliedVariants(EnumSet.of(GameType.Commander));

                Deck d1 = loadCommanderDeckByFilename(deckA);
                Deck d2 = loadCommanderDeckByFilename(deckB);

                List<RegisteredPlayer> players = new ArrayList<>();

                RegisteredPlayer p1 = RegisteredPlayer.forCommander(d1);
                p1.setPlayer(GamePlayerUtil.createAiPlayer("External-1", 0, 0));
                players.add(p1);

                RegisteredPlayer p2 = RegisteredPlayer.forCommander(d2);
                p2.setPlayer(GamePlayerUtil.createAiPlayer("AI-2", 0, 0));
                players.add(p2);

                Match m = new Match(rules, players, "rl_v0");
                currentMatch = m;

                Game g = m.createGame();
                currentGame = g;

                // swap controller for p1
                forge.game.player.Player gp1 = g.getPlayers().get(0);
                LobbyPlayer lp1 = gp1.getLobbyPlayer();
                GameContext ctx = currentCtx;
                gp1.dangerouslySetController(new ExternalController(g, gp1, lp1, ctx));

                // watch game over
                new Thread(() -> {
                    try {
                        while (currentGame == g && !g.isGameOver()) {
                            Thread.sleep(100);
                        }
                        if (currentGame == g && g.isGameOver()) {
                            doneFlag = true;
                            var outcome = g.getOutcome();
                            if (outcome != null) {
                                var w = outcome.getWinningLobbyPlayer();
                                winnerName = w != null ? w.getName() : null;
                            }
                        }
                    } catch (Throwable ignored) {
                    }
                }, "gameover-watcher").start();

                m.startGame(g);

                if (currentGame == g) {
                    doneFlag = true;
                    var outcome = g.getOutcome();
                    if (outcome != null) {
                        var w = outcome.getWinningLobbyPlayer();
                        winnerName = w != null ? w.getName() : null;
                    }
                }

            } catch (Throwable t) {
                // Forge AI can occasionally time out (java.util.concurrent.TimeoutException) when
                // choosing actions. Treat this as an opponent forfeit (External wins) so the env
                // stays stable for RL training, and avoid spamming stack traces.
                try {
                    String msg = String.valueOf(t);
                    if (t instanceof java.util.concurrent.TimeoutException || msg.contains("TimeoutException")) {
                        winnerName = "External-1";
                        doneFlag = true;
                        return;
                    }
                } catch (Throwable ignored) {}

                t.printStackTrace();
                doneFlag = true;
            }
        }, "forge-game-thread").start();
    }

    private static final class HeadlessGui implements IGuiBase {
        private final String assetsDir;
        HeadlessGui(String assetsDir) { this.assetsDir = assetsDir; }

        @Override public boolean isRunningOnDesktop() { return false; }
        @Override public boolean isLibgdxPort() { return false; }
        @Override public String getCurrentVersion() { return "rl_v0"; }
        @Override public String getAssetsDir() { return assetsDir; }
        @Override public ImageFetcher getImageFetcher() { return null; }
        @Override public void invokeInEdtNow(Runnable runnable) { runnable.run(); }
        @Override public void invokeInEdtLater(Runnable runnable) { runnable.run(); }
        @Override public void invokeInEdtAndWait(Runnable proc) { proc.run(); }
        @Override public boolean isGuiThread() { return false; }
        @Override public ISkinImage getSkinIcon(FSkinProp skinProp) { return null; }
        @Override public ISkinImage getUnskinnedIcon(String path) { return null; }
        @Override public ISkinImage getCardArt(PaperCard card) { return null; }
        @Override public ISkinImage getCardArt(PaperCard card, boolean backFace) { return null; }
        @Override public ISkinImage createLayeredImage(PaperCard card, FSkinProp background, String overlayFilename, float opacity) { return null; }
        @Override public void showBugReportDialog(String title, String text, boolean showExitAppBtn) { }
        @Override public void showImageDialog(ISkinImage image, String message, String title) { }
        @Override public int showOptionDialog(String message, String title, FSkinProp icon, List<String> options, int defaultOption) { return defaultOption; }
        @Override public String showInputDialog(String message, String title, FSkinProp icon, String initialInput, List<String> inputOptions, boolean isNumeric) { return initialInput; }
        @Override public <T> List<T> getChoices(String message, int min, int max, Collection<T> choices, Collection<T> selected, FSerializableFunction<T, String> display) { return new ArrayList<>(); }
        @Override public <T> List<T> order(String title, String top, int remainingObjectsMin, int remainingObjectsMax, List<T> sourceChoices, List<T> destChoices) { return new ArrayList<>(); }
        @Override public String showFileDialog(String title, String defaultDir) { return null; }
        @Override public File getSaveFile(File defaultFile) { return defaultFile; }
        @Override public void download(GuiDownloadService service, Consumer<Boolean> callback) { if (callback != null) callback.accept(false); }
        @Override public void refreshSkin() { }
        @Override public void showCardList(String title, String message, List<PaperCard> list) { }
        @Override public boolean showBoxedProduct(String title, String message, List<PaperCard> list) { return false; }
        @Override public PaperCard chooseCard(String title, String message, List<PaperCard> list) { return null; }
        @Override public int getAvatarCount() { return 0; }
        @Override public int getSleevesCount() { return 0; }
        @Override public void copyToClipboard(String text) { }
        @Override public void browseToUrl(String url) throws IOException, URISyntaxException { }
        @Override public boolean isSupportedAudioFormat(File file) { return false; }
        @Override public IAudioClip createAudioClip(String filename) { return null; }
        @Override public IAudioMusic createAudioMusic(String filename) { return null; }
        @Override public void startAltSoundSystem(String filename, boolean isSynchronized) { }
        @Override public void clearImageCache() { }
        @Override public void showSpellShop() { }
        @Override public void showBazaar() { }
        @Override public forge.gui.interfaces.IGuiGame getNewGuiGame() { return null; }
        @Override public HostedMatch hostMatch() { return null; }
        @Override public void runBackgroundTask(String message, Runnable task) { if (task != null) task.run(); }
        @Override public String encodeSymbols(String str, boolean formatReminderText) { return str; }
        @Override public void preventSystemSleep(boolean preventSleep) { }
        @Override public float getScreenScale() { return 1.0f; }
        @Override public UpnpServiceConfiguration getUpnpPlatformService() { return null; }
    }

    public static void main(String[] args) throws Exception {
        String assetsDir = System.getProperty("forge.assetsDir", new File(".").getAbsolutePath());
        GuiBase.setInterface(new HeadlessGui(assetsDir));
        System.setProperty("java.awt.headless", "true");

        int portTmp = 8799;
        try {
            portTmp = Integer.parseInt(System.getProperty("forge.port", "8799"));
        } catch (Throwable ignored) {}
        final int port = portTmp;

        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", port), 0);
        // HTTP executor thread pool: handles concurrent requests (/advance_wait, /act, /wait_ready, etc.).
        // With multi-env training we can have many simultaneous requests, so prefer a bounded pool.
        int httpThreads = 24;
        try { httpThreads = Integer.parseInt(System.getProperty("forge.httpThreads", "24")); } catch (Throwable ignored) {}
        server.setExecutor(java.util.concurrent.Executors.newFixedThreadPool(httpThreads));

        server.createContext("/health", ex -> write(ex, 200, json(Map.of("ok", true, "port", port))));

        server.createContext("/legal_actions", ex -> {
            try {
                write(ex, 200, json(Map.of("ok", true, "actions", legalActionsSnapshot())));
            } catch (Throwable t) {
                t.printStackTrace();
                write(ex, 200, json(Map.of("ok", false, "error", "exception_in_legal_actions", "actions", List.of("PASS"))));
            }
        });

        server.createContext("/wait_ready", ex -> {
            try {
                long deadline = System.currentTimeMillis() + 10000;
                while (System.currentTimeMillis() < deadline) {
                    Map<String, Object> s = snapshotState();
                    if (s.get("turn") != null && s.get("phase") != null) {
                        Map<String, Object> resp = new LinkedHashMap<>();
                        resp.put("ok", true);
                        resp.putAll(s);
                        resp.put("actions", legalActionsSnapshot());
                        Game g = currentGame;
                        resp.put("done", g != null && g.isGameOver());
                        write(ex, 200, json(resp));
                        return;
                    }
                    Thread.sleep(50);
                }
                Map<String, Object> resp = new LinkedHashMap<>();
                resp.put("ok", false);
                resp.put("error", "timeout waiting for ready");
                resp.putAll(snapshotState());
                write(ex, 200, json(resp));
            } catch (Throwable t) {
                t.printStackTrace();
                write(ex, 500, json(Map.of("ok", false, "error", "exception")));
            }
        });

        // Replay support: dump Forge GameLog as text
        server.createContext("/gamelog", ex -> {
            try {
                Game g = currentGame;
                if (g == null) {
                    write(ex, 409, json(Map.of("ok", false, "error", "no_game")));
                    return;
                }
                StringBuilder sb = new StringBuilder();
                try {
                    forge.game.GameLog gl = g.getGameLog();
                    if (gl != null) {
                        for (forge.game.GameLogEntryType t : forge.game.GameLogEntryType.values()) {
                            java.util.List<forge.game.GameLogEntry> es;
                            try { es = gl.getLogEntries(t); } catch (Throwable ignored) { es = null; }
                            if (es == null) continue;
                            for (forge.game.GameLogEntry e : es) {
                                if (e == null) continue;
                                sb.append('[').append(String.valueOf(t)).append("] ").append(String.valueOf(e.message)).append('\n');
                            }
                        }
                    }
                } catch (Throwable ignored) {}
                String txt = sb.toString();
                // Remove control chars that can break JSON parsing (keep \t \n \r)
                try { txt = txt.replaceAll("[\\x00-\\x08\\x0B\\x0C\\x0E-\\x1F]", " "); } catch (Throwable ignored) {}
                write(ex, 200, json(Map.of("ok", true, "text", txt)));
            } catch (Throwable t) {
                t.printStackTrace();
                write(ex, 500, json(Map.of("ok", false, "error", "exception_in_gamelog")));
            }
        });

        server.createContext("/reset", new HttpHandler() {
            @Override public void handle(HttpExchange ex) throws IOException {
                if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) {
                    write(ex, 405, json(Map.of("error", "POST only")));
                    return;
                }

                currentGame = null;
                currentMatch = null;

                // New context per reset so old blocked controller threads can't interfere.
                currentCtx = new GameContext();
                currentCtx.q.clear();

                lastP1Life = null;
                lastP2Life = null;
                lastP1Creatures = null;
                lastP2Creatures = null;

                startGameThread("krenko.dck", "stompy_goreclaw.dck");

                // Block briefly until game is actually ready (turn/phase available) to avoid races in multi-env.
                long deadline = System.currentTimeMillis() + 15000;
                while (System.currentTimeMillis() < deadline) {
                    Map<String, Object> s = snapshotState();
                    if (s.get("turn") != null && s.get("phase") != null) {
                        write(ex, 200, json(Map.of("ok", true, "message", "started")));
                        return;
                    }
                    try { Thread.sleep(25); } catch (InterruptedException ie) { Thread.currentThread().interrupt(); break; }
                }
                write(ex, 200, json(Map.of("ok", false, "error", "timeout_waiting_for_game_start")));
            }
        });

        // Advance the game until ExternalController needs an action (or game over).
        // Short-poll version (kept for compatibility).
        server.createContext("/advance", new HttpHandler() {
            @Override public void handle(HttpExchange ex) throws IOException {
                if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) {
                    write(ex, 405, json(Map.of("error", "POST only")));
                    return;
                }

                Game g = currentGame;
                GameContext ctx = currentCtx;
                long deadline = System.currentTimeMillis() + 300;
                while (g != null && !doneFlag && !g.isGameOver() && System.currentTimeMillis() < deadline) {
                    if (ctx != null && ctx.waiting.get()) {
                        break;
                    }
                    PhaseHandler ph = g.getPhaseHandler();
                    boolean isOurPriority = false;
                    if (ph != null && ph.getPriorityPlayer() != null) {
                        isOurPriority = "External-1".equals(ph.getPriorityPlayer().getName());
                    }
                    if (isOurPriority && ctx != null && ctx.q.isEmpty()) {
                        ctx.q.offer("PASS");
                    }
                    try { Thread.sleep(10); } catch (InterruptedException ie) { Thread.currentThread().interrupt(); break; }
                }

                boolean done = doneFlag || (g != null && g.isGameOver());
                Map<String, Object> resp = new LinkedHashMap<>();
                resp.put("ok", true);
                resp.putAll(snapshotState());
                resp.put("legal_actions", legalActionsSnapshot());
                resp.put("waiting", ctx != null && ctx.waiting.get());
                resp.put("done", done);
                if (winnerName != null) resp.put("winner", winnerName);
                write(ex, 200, json(resp));
            }
        });

        // Long-poll version: blocks until a *meaningful* decision point (or done/timeout).
        // Key optimization: if the only legal action is PASS, auto-enqueue PASS and keep advancing.
        server.createContext("/advance_wait", new HttpHandler() {
            @Override public void handle(HttpExchange ex) throws IOException {
                if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) {
                    write(ex, 405, json(Map.of("error", "POST only")));
                    return;
                }

                Game g = currentGame;
                GameContext ctx = currentCtx;
                long deadline = System.currentTimeMillis() + 5000;
                int autoPasses = 0;

                while (g != null && !doneFlag && !g.isGameOver() && System.currentTimeMillis() < deadline) {
                    if (ctx != null && ctx.waiting.get()) {
                        List<String> acts = legalActionsSnapshot();
                        if (acts != null && acts.size() == 1 && "PASS".equalsIgnoreCase(String.valueOf(acts.get(0)))) {
                            // PASS-only window: not a real RL decision.
                            if (autoPasses < 200) {
                                ctx.q.clear();
                                ctx.q.offer("PASS");
                                autoPasses++;
                            } else {
                                break; // safety
                            }
                        } else {
                            break; // meaningful decision point
                        }
                    }

                    PhaseHandler ph = g.getPhaseHandler();
                    boolean isOurPriority = false;
                    if (ph != null && ph.getPriorityPlayer() != null) {
                        isOurPriority = "External-1".equals(ph.getPriorityPlayer().getName());
                    }
                    if (isOurPriority && ctx != null && ctx.q.isEmpty()) {
                        ctx.q.offer("PASS");
                    }
                    try { Thread.sleep(10); } catch (InterruptedException ie) { Thread.currentThread().interrupt(); break; }
                }

                boolean done = doneFlag || (g != null && g.isGameOver());
                Map<String, Object> resp = new LinkedHashMap<>();
                resp.put("ok", true);
                resp.putAll(snapshotState());
                resp.put("legal_actions", legalActionsSnapshot());
                resp.put("waiting", ctx != null && ctx.waiting.get());
                resp.put("done", done);
                if (winnerName != null) resp.put("winner", winnerName);
                write(ex, 200, json(resp));
            }
        });

        // Apply an action at a decision point (requires waiting=true).
        // Enqueues the action, then advances until the next decision point (or game over),
        // returning the post-action state + shaped reward.
        server.createContext("/act", new HttpHandler() {
            @Override public void handle(HttpExchange ex) throws IOException {
                try {
                    if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) {
                        write(ex, 405, json(Map.of("error", "POST only")));
                        return;
                    }

                    Game g = currentGame;
                    if (g == null) {
                        write(ex, 409, json(Map.of("ok", false, "error", "no_game")));
                        return;
                    }

                    String body = readBody(ex);
                    if (body == null) body = "";
                    java.util.regex.Matcher mid = java.util.regex.Pattern.compile("\\\"action_id\\\"\\s*:\\s*(\\d+)").matcher(body);
                    int actionId = 0;
                    if (mid.find()) actionId = Integer.parseInt(mid.group(1));

                    List<String> currentActs = legalActionsSnapshot();
                    if (actionId < 0 || actionId >= currentActs.size()) actionId = 0;
                    String actStr = currentActs.get(actionId);

                    GameContext ctx = currentCtx;
                    if (ctx == null || !ctx.waiting.get()) {
                        write(ex, 409, json(Map.of("ok", false, "error", "not_waiting")));
                        return;
                    }

                    Map<String, Object> snapBefore = snapshotState();

                    // enqueue action
                    ctx.q.clear();
                    ctx.q.offer(actStr);

                    // Don't block here waiting for the next decision.
                    // The Python env will call /advance after /act.
                    // We only wait a tiny amount to allow immediate effects to settle.
                    long deadline = System.currentTimeMillis() + 200;
                    while (!doneFlag && !g.isGameOver() && System.currentTimeMillis() < deadline) {
                        try { Thread.sleep(5); } catch (InterruptedException ie) { Thread.currentThread().interrupt(); break; }
                    }

                    Map<String, Object> snapNow = snapshotState();

                    double shaped = ("PASS".equalsIgnoreCase(actStr) ? 0.0 : -0.0001);
                    try {
                        Integer p1LifePrev = (Integer) snapBefore.get("p1_life");
                        Integer p2LifePrev = (Integer) snapBefore.get("p2_life");
                        Integer p1LifeNow = (Integer) snapNow.get("p1_life");
                        Integer p2LifeNow = (Integer) snapNow.get("p2_life");
                        Integer p1CreatPrev = (Integer) snapBefore.get("p1_creatures");
                        Integer p2CreatPrev = (Integer) snapBefore.get("p2_creatures");
                        Integer p1CreatNow = (Integer) snapNow.get("p1_creatures");
                        Integer p2CreatNow = (Integer) snapNow.get("p2_creatures");

                        if (p1LifePrev != null && p2LifePrev != null && p1LifeNow != null && p2LifeNow != null) {
                            int dmgToOpp = p2LifePrev - p2LifeNow;
                            int dmgToUs = p1LifePrev - p1LifeNow;
                            shaped += 0.01 * (dmgToOpp - dmgToUs);
                        }
                        if (p1CreatPrev != null && p2CreatPrev != null && p1CreatNow != null && p2CreatNow != null) {
                            int ourDelta = p1CreatNow - p1CreatPrev;
                            int oppDelta = p2CreatNow - p2CreatPrev;
                            shaped += 0.02 * (ourDelta - oppDelta);
                        }
                    } catch (Throwable ignored) {}

                    boolean done = doneFlag || g.isGameOver();
                    String winner = winnerName;
                    double reward;
                    if (done) {
                        if (winner == null) {
                            var outcome = g.getOutcome();
                            if (outcome != null) {
                                var w = outcome.getWinningLobbyPlayer();
                                winner = w != null ? w.getName() : null;
                            }
                        }
                        if (winner != null) reward = winner.contains("External") ? 1.0 : -1.0;
                        else reward = 0.0;
                        reward += shaped;
                    } else {
                        reward = shaped;
                    }

                    Map<String, Object> resp = new LinkedHashMap<>();
                    resp.put("ok", true);
                    resp.put("action_taken", actStr);
                    resp.put("action_id", actionId);
                    resp.put("reward", reward);
                    resp.putAll(snapshotState());
                    resp.put("legal_actions", legalActionsSnapshot());
                    resp.put("waiting", ctx.waiting.get());
                    resp.put("done", done);
                    if (winner != null) resp.put("winner", winner);
                    write(ex, 200, json(resp));

                } catch (Throwable t) {
                    t.printStackTrace();
                    write(ex, 500, json(Map.of("ok", false, "error", "exception_in_act", "message", String.valueOf(t))));
                }
            }
        });

        // Combined step+advance endpoint to reduce HTTP chatter
        server.createContext("/step_wait", new HttpHandler() {
            @Override public void handle(HttpExchange ex) throws IOException {
                try {
                    if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) {
                        write(ex, 405, json(Map.of("error", "POST only")));
                        return;
                    }

                    Game g = currentGame;
                    GameContext ctx = currentCtx;
                    // If a client hits us before /reset finished starting the game, wait briefly.
                    if (g == null || ctx == null) {
                        long d = System.currentTimeMillis() + 3000;
                        while (System.currentTimeMillis() < d) {
                            g = currentGame;
                            ctx = currentCtx;
                            Map<String, Object> s = snapshotState();
                            if (g != null && ctx != null && s.get("turn") != null && s.get("phase") != null) break;
                            try { Thread.sleep(25); } catch (InterruptedException ie) { Thread.currentThread().interrupt(); break; }
                        }
                        g = currentGame;
                        ctx = currentCtx;
                        if (g == null || ctx == null) {
                            write(ex, 409, json(Map.of("ok", false, "error", "no_game")));
                            return;
                        }
                    }

                    // parse action_id (optional)
                    String body = readBody(ex);
                    if (body == null) body = "";
                    java.util.regex.Matcher mid = java.util.regex.Pattern.compile("\\\"action_id\\\"\\s*:\\s*(\\d+)").matcher(body);
                    Integer actionId = null;
                    if (mid.find()) actionId = Integer.parseInt(mid.group(1));

                    // Advance until a meaningful decision point (or done)
                    long deadline = System.currentTimeMillis() + 5000;
                    int autoPasses = 0;
                    while (g != null && !doneFlag && !g.isGameOver() && System.currentTimeMillis() < deadline) {
                        if (ctx.waiting.get()) {
                            List<String> acts = legalActionsSnapshot();
                            if (acts != null && acts.size() == 1 && "PASS".equalsIgnoreCase(String.valueOf(acts.get(0)))) {
                                if (autoPasses++ < 200) {
                                    ctx.q.clear();
                                    ctx.q.offer("PASS");
                                } else break;
                            } else {
                                break;
                            }
                        }
                        PhaseHandler ph = g.getPhaseHandler();
                        boolean isOurPriority = false;
                        if (ph != null && ph.getPriorityPlayer() != null) {
                            isOurPriority = "External-1".equals(ph.getPriorityPlayer().getName());
                        }
                        if (isOurPriority && ctx.q.isEmpty()) {
                            ctx.q.offer("PASS");
                        }
                        try { Thread.sleep(10); } catch (InterruptedException ie) { Thread.currentThread().interrupt(); break; }
                    }

                    Map<String, Object> snapBefore = snapshotState();
                    double shaped = 0.0;
                    String actStr = "PASS";
                    int usedActionId = 0;

                    if (!doneFlag && !g.isGameOver() && ctx.waiting.get()) {
                        List<String> acts = legalActionsSnapshot();
                        if (acts == null || acts.isEmpty()) acts = List.of("PASS");
                        if (actionId != null) {
                            usedActionId = Math.max(0, Math.min(actionId, acts.size() - 1));
                        }
                        actStr = acts.get(usedActionId);

                        // enqueue chosen action
                        ctx.q.clear();
                        ctx.q.offer(actStr);
                        shaped = ("PASS".equalsIgnoreCase(actStr) ? 0.0 : -0.0001);
                    }

                    // allow immediate effects to settle briefly
                    long settle = System.currentTimeMillis() + 200;
                    while (!doneFlag && !g.isGameOver() && System.currentTimeMillis() < settle) {
                        try { Thread.sleep(5); } catch (InterruptedException ie) { Thread.currentThread().interrupt(); break; }
                    }

                    // Advance again to next meaningful decision point
                    deadline = System.currentTimeMillis() + 5000;
                    autoPasses = 0;
                    while (g != null && !doneFlag && !g.isGameOver() && System.currentTimeMillis() < deadline) {
                        if (ctx.waiting.get()) {
                            List<String> acts = legalActionsSnapshot();
                            if (acts != null && acts.size() == 1 && "PASS".equalsIgnoreCase(String.valueOf(acts.get(0)))) {
                                if (autoPasses++ < 200) {
                                    ctx.q.clear();
                                    ctx.q.offer("PASS");
                                } else break;
                            } else {
                                break;
                            }
                        }
                        PhaseHandler ph = g.getPhaseHandler();
                        boolean isOurPriority = false;
                        if (ph != null && ph.getPriorityPlayer() != null) {
                            isOurPriority = "External-1".equals(ph.getPriorityPlayer().getName());
                        }
                        if (isOurPriority && ctx.q.isEmpty()) {
                            ctx.q.offer("PASS");
                        }
                        try { Thread.sleep(10); } catch (InterruptedException ie) { Thread.currentThread().interrupt(); break; }
                    }

                    Map<String, Object> snapNow = snapshotState();

                    // reward shaping: life+creature deltas
                    try {
                        Integer p1LifePrev = (Integer) snapBefore.get("p1_life");
                        Integer p2LifePrev = (Integer) snapBefore.get("p2_life");
                        Integer p1LifeNow = (Integer) snapNow.get("p1_life");
                        Integer p2LifeNow = (Integer) snapNow.get("p2_life");
                        Integer p1CreatPrev = (Integer) snapBefore.get("p1_creatures");
                        Integer p2CreatPrev = (Integer) snapBefore.get("p2_creatures");
                        Integer p1CreatNow = (Integer) snapNow.get("p1_creatures");
                        Integer p2CreatNow = (Integer) snapNow.get("p2_creatures");

                        if (p1LifePrev != null && p2LifePrev != null && p1LifeNow != null && p2LifeNow != null) {
                            int dmgToOpp = p2LifePrev - p2LifeNow;
                            int dmgToUs = p1LifePrev - p1LifeNow;
                            shaped += 0.01 * (dmgToOpp - dmgToUs);
                        }
                        if (p1CreatPrev != null && p2CreatPrev != null && p1CreatNow != null && p2CreatNow != null) {
                            int ourDelta = p1CreatNow - p1CreatPrev;
                            int oppDelta = p2CreatNow - p2CreatPrev;
                            shaped += 0.02 * (ourDelta - oppDelta);
                        }
                    } catch (Throwable ignored) {}

                    boolean done = doneFlag || g.isGameOver();
                    String winner = winnerName;
                    double reward;
                    if (done) {
                        if (winner == null) {
                            var outcome = g.getOutcome();
                            if (outcome != null) {
                                var w = outcome.getWinningLobbyPlayer();
                                winner = w != null ? w.getName() : null;
                            }
                        }
                        if (winner != null) reward = winner.contains("External") ? 1.0 : -1.0;
                        else reward = 0.0;
                        reward += shaped;
                    } else {
                        reward = shaped;
                    }

                    Map<String, Object> resp = new LinkedHashMap<>();
                    resp.put("ok", true);
                    resp.put("action_taken", actStr);
                    resp.put("action_id", usedActionId);
                    resp.put("reward", reward);
                    resp.putAll(snapshotState());
                    resp.put("legal_actions", legalActionsSnapshot());
                    resp.put("waiting", ctx.waiting.get());
                    resp.put("done", done);
                    if (winner != null) resp.put("winner", winner);
                    write(ex, 200, json(resp));

                } catch (Throwable t) {
                    t.printStackTrace();
                    write(ex, 500, json(Map.of("ok", false, "error", "exception_in_step_wait", "message", String.valueOf(t))));
                }
            }
        });

        server.createContext("/step", new HttpHandler() {
            @Override public void handle(HttpExchange ex) throws IOException {
                try {
                    if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) {
                        write(ex, 405, json(Map.of("error", "POST only")));
                        return;
                    }

                    String body = readBody(ex);
                    if (body == null) body = "";
                    java.util.regex.Matcher mid = java.util.regex.Pattern.compile("\\\"action_id\\\"\\s*:\\s*(\\d+)").matcher(body);
                    int actionId = 0;
                    if (mid.find()) actionId = Integer.parseInt(mid.group(1));

                    List<String> currentActs = legalActionsSnapshot();
                    if (actionId < 0 || actionId >= currentActs.size()) actionId = 0;
                    String actStr = currentActs.get(actionId);

                    Game g = currentGame;
                    boolean isOurPriority = false;
                    if (g != null) {
                        PhaseHandler ph = g.getPhaseHandler();
                        if (ph != null && ph.getPriorityPlayer() != null) {
                            isOurPriority = "External-1".equals(ph.getPriorityPlayer().getName());
                        }
                    }

                    boolean enqueued = false;
                    Map<String, Object> snapBefore = snapshotState();

                    GameContext ctx = currentCtx;
                    if (isOurPriority && ctx != null) {
                        if (!ctx.q.offer(actStr)) {
                            ctx.q.clear();
                            ctx.q.offer(actStr);
                        }
                        enqueued = true;
                    }

                    // Compute reward: only apply shaping when our action was enqueued.
                    Map<String, Object> snapNow = snapshotState();
                    double shaped = 0.0;
                    if (enqueued) {
                        Integer p1LifeNow = (Integer) snapNow.get("p1_life");
                        Integer p2LifeNow = (Integer) snapNow.get("p2_life");
                        Integer p1CreatNow = (Integer) snapNow.get("p1_creatures");
                        Integer p2CreatNow = (Integer) snapNow.get("p2_creatures");

                        double stepPenalty = ("PASS".equalsIgnoreCase(actStr) ? 0.0 : -0.0001);
                        shaped += stepPenalty;

                        try {
                            Integer p1LifePrev = (Integer) snapBefore.get("p1_life");
                            Integer p2LifePrev = (Integer) snapBefore.get("p2_life");
                            Integer p1CreatPrev = (Integer) snapBefore.get("p1_creatures");
                            Integer p2CreatPrev = (Integer) snapBefore.get("p2_creatures");
                            if (p1LifePrev != null && p2LifePrev != null && p1LifeNow != null && p2LifeNow != null) {
                                int dmgToOpp = p2LifePrev - p2LifeNow;
                                int dmgToUs = p1LifePrev - p1LifeNow;
                                shaped += 0.01 * (dmgToOpp - dmgToUs);
                            }
                            if (p1CreatPrev != null && p2CreatPrev != null && p1CreatNow != null && p2CreatNow != null) {
                                int ourDelta = p1CreatNow - p1CreatPrev;
                                int oppDelta = p2CreatNow - p2CreatPrev;
                                shaped += 0.02 * (ourDelta - oppDelta);
                            }
                        } catch (Throwable ignored) {}

                        lastP1Life = p1LifeNow;
                        lastP2Life = p2LifeNow;
                        lastP1Creatures = p1CreatNow;
                        lastP2Creatures = p2CreatNow;
                    }

                    boolean done = doneFlag || (g != null && g.isGameOver());
                    String winner = winnerName;

                    double reward;
                    if (done && g != null) {
                        if (winner == null) {
                            var outcome = g.getOutcome();
                            if (outcome != null) {
                                var w = outcome.getWinningLobbyPlayer();
                                winner = w != null ? w.getName() : null;
                            }
                        }
                        if (winner != null) {
                            reward = winner.contains("External") ? 1.0 : -1.0;
                        } else {
                            reward = 0.0;
                        }
                        reward += shaped;
                    } else {
                        reward = shaped;
                    }

                    Map<String, Object> resp = new LinkedHashMap<>();
                    resp.put("ok", true);
                    resp.put("action_taken", actStr);
                    resp.put("action_id", actionId);
                    resp.put("enqueued", enqueued);
                    resp.putAll(snapshotState());
                    resp.put("legal_actions", legalActionsSnapshot());
                    resp.put("done", done);
                    resp.put("reward", reward);
                    if (winner != null) resp.put("winner", winner);

                    write(ex, 200, json(resp));
                } catch (Throwable t) {
                    t.printStackTrace();
                    write(ex, 500, json(Map.of("ok", false, "error", "exception_in_step", "message", String.valueOf(t))));
                }
            }
        });

        server.start();
        System.out.println("ForgeEnvServer listening on http://127.0.0.1:" + port);
    }
}
