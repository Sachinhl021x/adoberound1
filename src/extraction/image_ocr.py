"""Image OCR and analysis using Amazon Nova Pro LLM."""

import os
import base64
from typing import Optional, Any


class ImageOCRProcessor:
    """Process images and extract text using Amazon Nova Pro."""

    def __init__(self, model_id: str = None, region: str = None):
        """Initialize image OCR processor.

        Args:
            model_id: Bedrock model ID for image analysis (default: Nova Pro)
            region: AWS region
        """
        self.model_id = model_id or os.getenv("IMAGE_LLM_MODEL", "amazon.nova-pro-v1:0")
        self.region = region or os.getenv("AWS_REGION", "us-east-1")

        # Initialize Bedrock client
        self._init_bedrock_client()

    def _init_bedrock_client(self):
        """Initialize AWS Bedrock runtime client."""
        try:
            import boto3

            self.bedrock_runtime = boto3.client(
                "bedrock-runtime",
                region_name=self.region
            )
            print(f"✓ Initialized Bedrock for image analysis: {self.model_id}")
        except Exception as e:
            print(f"⚠ Failed to initialize Bedrock: {e}")
            self.bedrock_runtime = None

    def extract_text_from_image(self, image: Any) -> str:
        """Extract text from image object (Docling format).

        Args:
            image: Image object from Docling

        Returns:
            Extracted text description
        """
        # This is a placeholder - implement based on Docling's image format
        return self._analyze_image_with_llm(None, "Describe this image in detail, focusing on any text content.")

    def extract_text_from_image_bytes(self, image_bytes: bytes) -> str:
        """Extract text from image bytes using Nova Pro.

        Args:
            image_bytes: Raw image bytes

        Returns:
            Extracted text and description
        """
        if not self.bedrock_runtime:
            return "[Image: OCR not available]"

        # Encode image to base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        prompt = """Analyze this image and extract ALL text content.

Instructions:
1. Transcribe any text visible in the image (OCR)
2. Describe charts, graphs, or diagrams
3. Explain tables or structured data
4. Note the layout and organization

Provide a comprehensive description that captures all information."""

        return self._analyze_image_with_llm(image_base64, prompt)

    def _analyze_image_with_llm(self, image_base64: Optional[str], prompt: str) -> str:
        """Call Nova Pro to analyze image.

        Args:
            image_base64: Base64 encoded image
            prompt: Analysis prompt

        Returns:
            LLM response
        """
        if not self.bedrock_runtime:
            return "[Image analysis not available - Bedrock not initialized]"

        try:
            import json

            # Build request body for Nova Pro
            content = []

            # Add image if provided
            if image_base64:
                content.append({
                    "image": {
                        "format": "png",  # Adjust based on actual format
                        "source": {
                            "bytes": image_base64
                        }
                    }
                })

            # Add text prompt
            content.append({
                "text": prompt
            })

            request_body = {
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                "inferenceConfig": {
                    "max_new_tokens": 1000,
                    "temperature": 0.1,
                }
            }

            # Invoke model
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )

            # Parse response
            response_body = json.loads(response['body'].read())

            # Extract text from response (format varies by model)
            if 'content' in response_body:
                # Nova format
                for item in response_body['content']:
                    if 'text' in item:
                        return item['text']

            # Fallback
            return str(response_body)

        except Exception as e:
            print(f"⚠ Error analyzing image with LLM: {e}")
            return f"[Image analysis error: {str(e)}]"

    def extract_text_from_file(self, image_path: str) -> str:
        """Extract text from image file.

        Args:
            image_path: Path to image file

        Returns:
            Extracted text
        """
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            return self.extract_text_from_image_bytes(image_bytes)
        except Exception as e:
            print(f"⚠ Error reading image file: {e}")
            return f"[Error reading image: {str(e)}]"
