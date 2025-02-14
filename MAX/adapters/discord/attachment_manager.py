from typing import List, Dict, Optional
import aiohttp
from MAX.types import DiscordAttachment, ProcessedContent
from MAX.utils.logger import Logger


class AttachmentManager:
    def __init__(self, runtime):
        self.runtime = runtime
        self.logger = Logger()
        self._cache: Dict[str, DiscordAttachment] = {}

    async def process_message_content(
        self, content: str, raw_attachments: List[Dict]
    ) -> ProcessedContent:
        try:
            processed_attachments = []
            for attachment in raw_attachments:
                processed = await self.process_attachment(attachment)
                if processed:
                    processed_attachments.append(
                        {
                            "id": processed.id,
                            "url": processed.url,
                            "text": processed.text_content,
                            "description": processed.description,
                            "type": processed.content_type,
                        }
                    )

            return ProcessedContent(
                text=content, attachments=processed_attachments
            )
        except Exception as e:
            self.logger.error(f"Error processing message content: {str(e)}")
            return ProcessedContent(text=content)

    async def process_attachment(
        self, raw_attachment: Dict
    ) -> Optional[DiscordAttachment]:
        """Process individual attachment with caching"""
        cache_key = raw_attachment["url"]
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            attachment = DiscordAttachment(
                id=raw_attachment["id"],
                url=raw_attachment["url"],
                filename=raw_attachment["filename"],
                content_type=raw_attachment.get("content_type"),
                size=raw_attachment.get("size", 0),
            )

            if attachment.content_type:
                if attachment.content_type.startswith("text/"):
                    await self._process_text(attachment)
                elif attachment.content_type.startswith("application/pdf"):
                    await self._process_pdf(attachment)
                elif attachment.content_type.startswith("image/"):
                    await self._process_image(attachment)

            self._cache[cache_key] = attachment
            return attachment

        except Exception as e:
            self.logger.error(f"Error processing attachment: {str(e)}")
            return None

    async def _process_text(self, attachment: DiscordAttachment):
        async with aiohttp.ClientSession() as session:
            async with session.get(attachment.url) as response:
                text = await response.text()
                attachment.text_content = text
                attachment.description = (
                    f"Text content from {attachment.filename}"
                )

    async def _process_pdf(self, attachment: DiscordAttachment):
        if hasattr(self.runtime, "pdf_service"):
            async with aiohttp.ClientSession() as session:
                async with session.get(attachment.url) as response:
                    pdf_data = await response.read()
                    text = await self.runtime.pdf_service.process_pdf(pdf_data)
                    attachment.text_content = text
                    attachment.description = (
                        f"Extracted text from PDF: {attachment.filename}"
                    )

    async def _process_image(self, attachment: DiscordAttachment):
        if hasattr(self.runtime, "image_service"):
            description = await self.runtime.image_service.describe_image(
                attachment.url
            )
            attachment.description = description
