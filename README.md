# Image Fraud Detection API

A FastAPI-based service that provides comprehensive image analysis capabilities including fraud detection, EXIF metadata analysis, and image classification using Google Cloud Vision API and Vertex AI.

## Features

- **Web Search Detection**: Identifies potential fraud by searching for matching images across the web
- **EXIF Analysis**: Extracts and analyzes EXIF metadata to detect potential image manipulation
- **Image Classification**: Uses Google Cloud Vertex AI to classify image content
- **Base64 Image Support**: Handles base64 encoded images for easy integration
- **Built-in Validation**: Automatic image validation including size and format checks
- **Comprehensive Error Handling**: Detailed error messages and logging

## Prerequisites

- Python 3.10+
- Google Cloud Platform account with Vision API and Vertex AI enabled
- Docker (for containerized deployment)
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Google Cloud credentials:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
```

## Running the API

### Local Development
```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

### Using Docker
```bash
# Build the container
docker build -t image-fraud-detection .

# Run the container
docker run -p 8080:8080 image-fraud-detection
```

## API Usage

### Endpoint

`POST /api/v1.1/analyze`

### Request Format

```json
{
    "source_type": "base64",
    "source": "<base64-encoded-image>",
    "analysis_type": "web_search" | "exif" | "classification"
}
```

### Analysis Types

1. **web_search**: Searches for matching images across the web
   - Returns matching image counts and URLs
   - Indicates if the image appears elsewhere online

2. **exif**: Analyzes image metadata
   - Camera model
   - Software used for editing
   - Original and digitized timestamps
   - Metadata manipulation warnings

3. **classification**: Uses Vertex AI for image classification
   - Provides confidence scores
   - Multiple category predictions
   - Model version information

### Example Response

```json
{
    "web_search": {
        "is_fraud": false,
        "matching_images_count": 0,
        "full_matching_images": [],
        "partial_matching_images": []
    }
}
```

## Configuration

Key configurations are managed through the `VisionConfig` class in `utils/fraud_utils.py`:

- `MAX_IMAGE_SIZE`: 1.5MB (1,500,000 bytes)
- `PROJECT`: GCP project ID
- `LOCATION`: GCP region
- `ENDPOINT_ID`: Vertex AI endpoint ID

## Error Handling

The API implements comprehensive error handling:

- 400: Bad Request (invalid input)
- 500: Internal Server Error (unexpected issues)
- Detailed error messages in response

## Logging

Comprehensive logging is implemented throughout the application:
- Request details
- Processing steps
- Error information
- Performance metrics

## Project Structure

```
├── app.py                 # FastAPI application
├── utils/
│   ├── fraud_utils.py    # Core analysis functionality
│   └── sample_request.py # Example usage
├── Dockerfile            # Container configuration
└── requirements.txt      # Python dependencies
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
