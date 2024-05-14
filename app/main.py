from fastapi import FastAPI

from app.iris.views import router as iris_router

app = FastAPI()


@app.get("/")
def health_check():
    """Health check endpoint"""
    return {"status": "ok"}


app.include_router(iris_router, prefix="/iris", tags=["Iris"])
