from fastapi import Depends, FastAPI, Response, Security
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi.middleware import SlowAPIASGIMiddleware

from app.endpoints import audio, chat, chunks, collections, completions, documents, embeddings, files, models, search
from app.helpers._metricsmiddleware import MetricsMiddleware
from app.schemas.security import User
from app.utils.config import APP_CONTACT_EMAIL, APP_CONTACT_URL, APP_DESCRIPTION, APP_VERSION
from app.utils.lifespan import lifespan
from app.utils.security import check_admin_api_key, check_api_key

app = FastAPI(
    title="Albert API",
    version=APP_VERSION,
    description=APP_DESCRIPTION,
    contact={"url": APP_CONTACT_URL, "email": APP_CONTACT_EMAIL},
    licence_info={"name": "MIT License", "identifier": "MIT"},
    lifespan=lifespan,
    docs_url="/swagger",
    redoc_url="/documentation",
)

# Prometheus metrics
# @TODO: env_var_name="ENABLE_METRICS"
app.instrumentator = Instrumentator().instrument(app=app)

# Middlewares
app.add_middleware(middleware_class=SlowAPIASGIMiddleware)
app.add_middleware(middleware_class=MetricsMiddleware)


# Monitoring
@app.get(path="/health", tags=["Monitoring"])
def health(user: User = Security(dependency=check_api_key)) -> Response:
    """
    Health check.
    """

    return Response(status_code=200)


app.instrumentator.expose(app=app, should_gzip=True, tags=["Monitoring"], dependencies=[Depends(dependency=check_admin_api_key)])

# Core
app.include_router(router=models.router, tags=["Core"], prefix="/v1")
app.include_router(router=chat.router, tags=["Core"], prefix="/v1")
app.include_router(router=completions.router, tags=["Core"], prefix="/v1")
app.include_router(router=embeddings.router, tags=["Core"], prefix="/v1")
app.include_router(router=audio.router, tags=["Core"], prefix="/v1")

# RAG
app.include_router(router=search.router, tags=["Retrieval Augmented Generation"], prefix="/v1")
app.include_router(router=collections.router, tags=["Retrieval Augmented Generation"], prefix="/v1")
app.include_router(router=files.router, tags=["Retrieval Augmented Generation"], prefix="/v1")
app.include_router(router=documents.router, tags=["Retrieval Augmented Generation"], prefix="/v1")
app.include_router(router=chunks.router, tags=["Retrieval Augmented Generation"], prefix="/v1")
