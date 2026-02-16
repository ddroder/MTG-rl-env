# RL v0 (Forge)

Goal: first end-to-end wiring for RL-style control.

What v0 does:
- Runs a 1v1 Commander game in Forge.
- Player A is controlled externally via a tiny HTTP server.
- For now the **only legal action** exposed is `PASS` (so Player A will mostly lose). This is intentional: it proves the loop.

Next step (v0.1): expose real legal actions (cast/activate/attack) + action masking.

## Prereqs
- Forge desktop jar: `../forge_dist/forge-gui-desktop-2.0.09-jar-with-dependencies.jar`
- Java 17 (we ship `../jre17` and `../jdk17` already)
- Deck files in `~/.forge/decks/commander/` (we currently use `krenko.dck` and `stompy_goreclaw.dck`)

## Build
```bash
cd /home/ddroder/users/dan/forge_env/rl_v0
../jdk17/bin/javac -cp ../forge_dist/forge-gui-desktop-2.0.09-jar-with-dependencies.jar -d build java/src/rl/ForgeEnvServer.java
../jdk17/bin/jar --create --file build/forge-env-server.jar -C build .
```

## Run server
```bash
cd /home/ddroder/users/dan/forge_env/rl_v0
../jre17/bin/java \
  -Xmx6G \
  -Djava.awt.headless=true \
  -Dforge.assetsDir=/home/ddroder/users/dan/forge_env/forge_dist \
  -cp build/forge-env-server.jar:/home/ddroder/users/dan/forge_env/forge_dist/forge-gui-desktop-2.0.09-jar-with-dependencies.jar \
  rl.ForgeEnvServer
```

Server listens on `http://127.0.0.1:8799`.

## Try it
```bash
curl -sS http://127.0.0.1:8799/health
curl -sS -XPOST http://127.0.0.1:8799/reset
curl -sS -XPOST http://127.0.0.1:8799/step -H 'content-type: application/json' -d '{"action":"PASS"}'
```

## TensorBoard
The python side will log to `runs/` once we start training.
