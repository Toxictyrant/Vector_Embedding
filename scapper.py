from langchain.document_loaders.text import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import re
import numpy as np

file_path = "Viratkholi.txt"

try:
    loader = TextLoader(file_path, encoding="utf-8")
    data = loader.load()

    # Convert data to a string if it's not already
    if not isinstance(data, str):
        data = str(data)

except Exception as e:
    print(f"Error loading {file_path}: {e}")

splitter = CharacterTextSplitter(separator=" ",
                                 chunk_size=200,
                                 chunk_overlap=3)

chunks = splitter.split_text(data)
encoder = SentenceTransformer("all-mpnet-base-v2")
vectors = encoder.encode(chunks)
dim = vectors.shape[1]
index = faiss.IndexFlatL2(dim)

search = "Name virat kholi's family members"
svec = np.array(encoder.encode(search)).reshape(1, -1)

index.add(vectors)
distance, I = index.search(svec,k=2)
input_string = str(I[0])
list = re.split(r'\s+|\[|\]', input_string)


print(list)
print(chunks[int(list[1])],chunks[int(list[2])])




