# Agentic Drone

This fork packages TypeFly so it runs smoothly on my mac
## Getting Started

1. **Install dependencies**
   ```bash
   make typefly
   ```
   The first run installs Python requirements, generates protobuf stubs, and launches the Gradio UI pointed at the virtual robot wrapper.

2. **Start the YOLO service**
   ```bash
   make SERVICE=yolo build
   ```
   The Makefile builds an ARM64 Docker image (based on `python:3.11-slim`) and runs it with the required gRPC ports exposed.

3. **Launch the web UI**
   ```bash
   make typefly
   ```
   Visit `http://localhost:50001` in a browser to see the camera stream and chat interface.

## Notes

- The YOLO container caches model weights after the first download. Watch progress:
  ```bash
  docker logs -f typefly-yolo
  ```
- On Apple silicon the container publishes ports `50050-50052` instead of relying on host networking, so the web UI can reach YOLO without emulation.
- To stop services:
  ```bash
  make SERVICE=yolo stop
  ```
