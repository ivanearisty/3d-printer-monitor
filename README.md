# Minimal 3D Print Failure Detection

A stripped-down version of the Obico server focused **only** on:
1. **Detecting 3D print failures** from an RTSP camera feed
2. **Stopping the printer** when failures are detected

No web interface, no accounts, no database, no notifications bloat.

## Architecture

```
┌─────────────────┐     frames      ┌──────────────┐
│  RTSP Camera    │ ───────────────▶│   Stream     │
│  (3D Printer)   │                 │   Analyzer   │
└─────────────────┘                 └──────┬───────┘
                                           │
                                    base64 │ image
                                           ▼
                                    ┌──────────────┐
                                    │   ML API     │
                                    │  (Detection) │
                                    └──────┬───────┘
                                           │
                                   failure │ detected
                                           ▼
                                    ┌──────────────┐
                                    │   Printer    │
                                    │   (Stop!)    │
                                    └──────────────┘
```

## Quick Start

### 1. Copy and configure environment

```bash
cp .env.example .env
# Edit .env with your camera and printer settings
```

### 2. Start the services

```bash
docker compose up -d
```

### 3. Check logs

```bash
# Watch all logs
docker compose logs -f

# Just the analyzer
docker compose logs -f stream_analyzer
```

## Configuration

### Camera Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `CAMERA_HOST` | IP address of your camera | `192.168.1.100` |
| `CAMERA_PORT` | RTSP port | `554` |
| `CAMERA_USERNAME` | Camera login | `admin` |
| `CAMERA_PASSWORD` | Camera password | - |
| `CAMERA_STREAM_PATH` | Stream path | `stream1` |

### Detection Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `DETECTION_THRESHOLD` | Confidence threshold (0.0-1.0). Lower = more sensitive | `0.40` |
| `ANALYSIS_INTERVAL_SECONDS` | How often to check frames | `10` |
| `CONSECUTIVE_FAILURES_TO_STOP` | Number of consecutive detections before stopping | `3` |

### Printer Control

| Variable | Description | Example |
|----------|-------------|---------|
| `PRINTER_STOP_WEBHOOK` | URL to call to stop the printer | See below |
| `PRINTER_API_KEY` | API key (OctoPrint, etc.) | - |

#### Supported Printer Interfaces

**OctoPrint:**
```
PRINTER_STOP_WEBHOOK=http://octopi.local/api/job
PRINTER_API_KEY=your_octoprint_api_key
```

**Moonraker (Klipper):**
```
PRINTER_STOP_WEBHOOK=http://klipper.local:7125/printer/emergency_stop
```

**Home Assistant:**
```
PRINTER_STOP_WEBHOOK=http://homeassistant.local:8123/api/webhook/stop_3d_printer
```

## TrueNAS Scale Deployment

1. Create a new app from the **Custom App** option
2. Use the `docker-compose.yml` or configure manually:
   - **ML API container**: Build from `./ml_api/Dockerfile`, port 3333
   - **Analyzer container**: Build from `./stream_analyzer/Dockerfile`
3. Add environment variables in the TrueNAS app config
4. Mount a host path to `/app/snapshots` to save detection images

Alternatively, use Portainer or similar to deploy the compose stack directly.

## Resource Requirements

- **ML API**: ~500MB RAM (model loading), low CPU except during inference
- **Stream Analyzer**: ~100MB RAM, low CPU

Total: **~1GB RAM** recommended

## Testing the ML API

```bash
# Health check
curl http://localhost:3333/hc/

# Test with an image URL
curl "http://localhost:3333/p/?img=https://example.com/print.jpg"

# Test with a local image
curl -X POST http://localhost:3333/p/ \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$(base64 -i test.jpg)\"}"
```

## Snapshots

Detection snapshots are saved to `./snapshots/` with annotations showing:
- Bounding boxes around detected failures
- Confidence scores
- Timestamp

## Troubleshooting

### Camera not connecting
- Verify RTSP URL: `ffplay rtsp://user:pass@host:554/stream1`
- Check firewall allows port 554
- Some cameras need specific paths like `/h264/ch1/main/av_stream`

### False positives
- Increase `DETECTION_THRESHOLD` (e.g., 0.50 or 0.60)
- Increase `CONSECUTIVE_FAILURES_TO_STOP`
- Ensure good lighting on the print

### Missed failures
- Decrease `DETECTION_THRESHOLD` (e.g., 0.30)
- Decrease `ANALYSIS_INTERVAL_SECONDS` for more frequent checks

## Credits

Uses the failure detection model from [Obico](https://github.com/TheSpaghettiDetective/obico-server).
