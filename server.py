"""
More pseudocode. 
"""

from fastapi import FastAPI, HTTPException
from typing import List

app = FastAPI()


@app.post("/add-doc/")
async def add_doc(file_path: str):
    # Stub for adding a document
    # TODO - Vectorize document in chunks
    # TODO - Store vectorized document
    # TODO - Store document metadata
    return {"message": "Document added successfully", "file_path": file_path}


@app.post("/add-image/")
async def add_image(file_path: str):
    # Stub for adding an image
    # TODO - Vectorize image
    # TODO - Store vectorized image
    # TODO - Store image metadata
    return {"message": "Image added successfully", "file_path": file_path}


@app.delete("/remove/")
async def remove(file_path: str):
    # Stub for removing a file
    # TODO - Remove vectorized document
    # TODO - Remove document metadata
    # TODO - Remove vectorized image
    # TODO - Remove image metadata
    # TODO - if no data don't break
    return {"message": "File removed successfully", "file_path": file_path}


@app.get("/search/")
async def search(k: int):
    # Stub for search functionality
    # TODO - Search the vectors with cosine similarity,
    # if a doc part upscale for that doc accordingly
    results = [{"file_path": f"file_{i}", "type": "doc/image"} for i in range(k)]
    return {"results": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
