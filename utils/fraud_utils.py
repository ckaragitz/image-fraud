import base64
import io
from typing import Dict, Literal, Tuple
from dataclasses import dataclass
import logging

from PIL import Image
import piexif
from google.cloud import vision, aiplatform
from google.cloud.aiplatform.gapic.schema import predict
from google.api_core import exceptions as google_exceptions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type definitions
AnalysisType = Literal['web_search', 'classification', 'exif']

@dataclass
class VisionConfig:
    PROJECT: str = "ck-vertex"
    LOCATION: str = "us-central1"
    ENDPOINT_ID: str = "6057421763162144768"
    MAX_IMAGE_SIZE: int = 1_500_000  # 1.5MB in bytes

class ImageAnalyzer:
    def __init__(self):
        self._vision_client = None
        self._vertex_client = None

    @property
    def vision_client(self):
        if not self._vision_client:
            try:
                self._vision_client = vision.ImageAnnotatorClient()
                logger.info("Vision client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Vision client: {e}")
                raise
        return self._vision_client

    @property
    def vertex_client(self):
        if not self._vertex_client:
            try:
                self._vertex_client = aiplatform.gapic.PredictionServiceClient(
                    client_options={"api_endpoint": f"{VisionConfig.LOCATION}-aiplatform.googleapis.com"}
                )
                logger.info("Vertex client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Vertex client: {e}")
                raise
        return self._vertex_client

    def validate_image(self, base64_string: str) -> Tuple[bytes, Image.Image]:
        """Validate and decode base64 image, returning both bytes and PIL Image."""
        try:
            logger.info("Starting image validation")
            cleaned_string = base64_string.replace('\n', '').replace('\r', '').strip()
            image_data = base64.b64decode(cleaned_string)

            # Check file size
            size = len(image_data)
            logger.info(f"Image size: {size} bytes")
            if size > VisionConfig.MAX_IMAGE_SIZE:
                raise ValueError(f"Image size ({size} bytes) exceeds maximum allowed size of {VisionConfig.MAX_IMAGE_SIZE} bytes")

            # Validate image format
            try:
                pil_image = Image.open(io.BytesIO(image_data))
                pil_image.verify()
                pil_image = Image.open(io.BytesIO(image_data))  # Reopen after verify
                logger.info(f"Image validated successfully. Format: {pil_image.format}")
                return image_data, pil_image
            except Exception as e:
                logger.error(f"Image validation failed: {e}")
                raise ValueError(f"Invalid image format: {str(e)}")

        except Exception as e:
            logger.error(f"Image validation failed: {e}")
            raise ValueError(f"Invalid image data: {str(e)}")

    def process_web_detection(self, image_data: bytes) -> Dict:
        """Process image through Vision API's web detection."""
        try:
            logger.info("Starting web detection")
            image = vision.Image(content=image_data)
            web_detection = self.vision_client.web_detection(image=image).web_detection

            is_fraud = bool(web_detection.full_matching_images)
            matching_count = len(web_detection.full_matching_images) if is_fraud else 0

            logger.info(f"Web detection completed. Found {matching_count} matching images")
            return {
                "is_fraud": is_fraud,
                "matching_images_count": matching_count,
                "full_matching_images": [img.url for img in web_detection.full_matching_images],
                "partial_matching_images": [img.url for img in web_detection.partial_matching_images]
            }
        except Exception as e:
            logger.error(f"Web detection failed: {e}")
            raise ValueError(f"Web detection failed: {str(e)}")

    def classify_image(self, encoded_content: str) -> Dict:
        """Classify image using Vertex AI."""
        try:
            logger.info("Starting image classification")
            instance = predict.instance.ImageClassificationPredictionInstance(
                content=encoded_content,
            ).to_value()

            parameters = predict.params.ImageClassificationPredictionParams(
                confidence_threshold=0.1,
                max_predictions=10,
            ).to_value()

            endpoint = self.vertex_client.endpoint_path(
                project=VisionConfig.PROJECT,
                location=VisionConfig.LOCATION,
                endpoint=VisionConfig.ENDPOINT_ID
            )

            response = self.vertex_client.predict(
                endpoint=endpoint,
                instances=[instance],
                parameters=parameters
            )

            predictions = []
            for pred in response.predictions:
                pred_dict = dict(pred)
                pred_dict['confidences'] = list(pred_dict['confidences'])
                pred_dict['displayNames'] = list(pred_dict['displayNames'])
                pred_dict['ids'] = list(pred_dict['ids'])
                predictions.append(pred_dict)

            logger.info("Classification completed successfully")
            return {
                "deployed_model_id": response.deployed_model_id,
                "model_version_id": response.model_version_id,
                "model_display_name": response.model_display_name,
                "predictions": predictions
            }
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            raise ValueError(f"Classification failed: {str(e)}")

    def analyze_exif(self, pil_image: Image.Image) -> Dict:
        """Analyze EXIF metadata from PIL Image."""
        try:
            logger.info("Starting EXIF analysis")
            if 'exif' in pil_image.info:
                exif_data = piexif.load(pil_image.info['exif'])
            else:
                logger.info("No EXIF data found in image")
                return {
                    "camera_model": "",
                    "software": "",
                    "datetime_original": "",
                    "datetime_digitized": "",
                    "warnings": ["No EXIF data found in image"]
                }

            analysis = {
                "camera_model": exif_data.get('0th', {}).get(piexif.ImageIFD.Model, b'').decode('utf-8', 'ignore'),
                "software": exif_data.get('0th', {}).get(piexif.ImageIFD.Software, b'').decode('utf-8', 'ignore'),
                "datetime_original": exif_data.get('Exif', {}).get(piexif.ExifIFD.DateTimeOriginal, b'').decode('utf-8', 'ignore'),
                "datetime_digitized": exif_data.get('Exif', {}).get(piexif.ExifIFD.DateTimeDigitized, b'').decode('utf-8', 'ignore'),
                "warnings": []
            }

            if analysis["software"]:
                analysis["warnings"].append(f"Image edited with software: {analysis['software']}")
            if analysis["datetime_original"] and analysis["datetime_digitized"] and analysis["datetime_original"] != analysis["datetime_digitized"]:
                analysis["warnings"].append("Original date and digitized date do not match.")

            logger.info("EXIF analysis completed successfully")
            return analysis
        except Exception as e:
            logger.error(f"EXIF analysis failed: {e}")
            return {"error": f"EXIF analysis failed: {str(e)}"}

    def analyze(self, base64_content: str, analysis_type: AnalysisType) -> Dict:
        """Main analysis method."""
        try:
            logger.info(f"Starting analysis of type: {analysis_type}")
            
            # Validate image first
            image_data, pil_image = self.validate_image(base64_content)

            analysis_methods = {
                'web_search': lambda: self.process_web_detection(image_data),
                'classification': lambda: self.classify_image(base64_content),
                'exif': lambda: self.analyze_exif(pil_image)
            }

            if analysis_type not in analysis_methods:
                raise ValueError(f"Invalid analysis type: {analysis_type}")

            result = analysis_methods[analysis_type]()
            logger.info(f"Analysis completed successfully for type: {analysis_type}")
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise

# Create a singleton instance
analyzer = ImageAnalyzer()