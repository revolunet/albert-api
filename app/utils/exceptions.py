from fastapi import HTTPException

from app.utils.config import MAX_CONTENT_SIZE


# 400
class ParsingFileFailedException(HTTPException):
    def __init__(self, detail: str = "Parsing file failed.") -> None:
        super().__init__(status_code=400, detail=detail)


class NoChunksToUpsertException(HTTPException):
    def __init__(self, detail: str = "No chunks to upsert.") -> None:
        super().__init__(status_code=400, detail=detail)


class ModelNotAvailableException(HTTPException):
    def __init__(self, detail: str = "Model not available.") -> None:
        super().__init__(status_code=400, detail=detail)


# 403
class InvalidAuthenticationSchemeException(HTTPException):
    def __init__(self, detail: str = "Invalid authentication scheme.") -> None:
        super().__init__(status_code=403, detail=detail)


class InvalidAPIKeyException(HTTPException):
    def __init__(self, detail: str = "Invalid API key.") -> None:
        super().__init__(status_code=403, detail=detail)


class InsufficientRightsException(HTTPException):
    def __init__(self, detail: str = "Insufficient rights.") -> None:
        super().__init__(status_code=403, detail=detail)


# 404
class CollectionNotFoundException(HTTPException):
    def __init__(self, detail: str = "Collection not found.") -> None:
        super().__init__(status_code=404, detail=detail)


class ModelNotFoundException(HTTPException):
    def __init__(self, detail: str = "Model not found.") -> None:
        super().__init__(status_code=404, detail=detail)


# 413
class ContextLengthExceededException(HTTPException):
    def __init__(self, detail: str = "Context length exceeded.") -> None:
        super().__init__(status_code=413, detail=detail)


class FileSizeLimitExceededException(HTTPException):
    def __init__(self, detail: str = f"File size limit exceeded (max: {MAX_CONTENT_SIZE} bytes).") -> None:
        super().__init__(status_code=413, detail=detail)


# 422
class InvalidJSONFormatException(HTTPException):
    def __init__(self, detail: str = "Invalid JSON file format.") -> None:
        super().__init__(status_code=422, detail=detail)


class WrongModelTypeException(HTTPException):
    def __init__(self, detail: str = "Wrong model type.") -> None:
        super().__init__(status_code=422, detail=detail)


class MaxTokensExceededException(HTTPException):
    def __init__(self, detail: str = "Max tokens exceeded.") -> None:
        super().__init__(status_code=422, detail=detail)


class DifferentCollectionsModelsException(HTTPException):
    def __init__(self, detail: str = "Different collections models.") -> None:
        super().__init__(status_code=422, detail=detail)


class UnsupportedFileTypeException(HTTPException):
    def __init__(self, detail: str = "Unsupported file type.") -> None:
        super().__init__(status_code=422, detail=detail)
