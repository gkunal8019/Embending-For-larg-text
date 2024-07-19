# Import required libraries
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import TextSplitter  # Ensure this is imported if used
import text_splitter  # Ensure your actual text_splitter import matches

    def create_faiss_index(documents, batch_size, api_token):
        """
        Create a FAISS index from a list of documents using Hugging Face Hub embeddings.
    
        Args:
            documents (list): List of document texts.
            batch_size (int): Number of documents to process in each batch.
            api_token (str): Hugging Face Hub API token for accessing embeddings.
    
        Returns:
            FAISS: A FAISS index containing the documents.
        """
        # Function to split the documents into batches
        def split_into_batches(documents, batch_size):
            for i in range(0, len(documents), batch_size):
                yield documents[i:i + batch_size]
    
        # Initialize the embeddings model
        embeddings = HuggingFaceHubEmbeddings(huggingfacehub_api_token=api_token)
    
        # Create an empty FAISS database
        db = None
    
        # Process each batch
        for i, batch in enumerate(split_into_batches(documents, batch_size)):
            if db is None:
                # Create a new FAISS database with the first batch
                db = FAISS.from_documents(batch, embeddings)
            else:
                # Create a temporary FAISS database with the new batch
                temp_db = FAISS.from_documents(batch, embeddings)
                # Merge the temporary database with the existing FAISS database
                db.merge_from(temp_db)
    
        print(f"Total documents in FAISS index: {db.index.ntotal}")
        return db

# Instructions:
# 1. Install the required packages:
#    pip install langchain_community
#
# 2. Replace `your_huggingfacehub_api_token` with your actual Hugging Face Hub API token.
#
# 3. Ensure `text_splitter` and `state_of_the_union` are defined appropriately. Adjust the import statements based on your project structure.
#
# 4. Run the code to create a FAISS index from your documents.

# Example usage:
# Replace the following placeholder with your actual document and text splitter setup
texts = text_splitter.create_documents([state_of_the_union])  # Replace with your actual documents
batch_size = 20
api_token = "your_huggingfacehub_api_token"  # Replace with your actual API token

# Create FAISS index
faiss_db = create_faiss_index(texts, batch_size, api_token)
