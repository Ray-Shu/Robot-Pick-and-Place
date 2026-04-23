## starting the docker

```bash
sudo docker compose up -d
```

## getting into the docker container
```bash
sudo docker exec -it robot-visual-servoing bash
```

## prepare for rviz - do in normal terminal
```bash
xhost +
```

## starting the code
```bash
python3 scripts/visual_servoing.py && tmux attach
```

## to kill
```bash
tmux kill-session
```

#### do this in pc terminal not docker
```bash
sudo docker compose down
```