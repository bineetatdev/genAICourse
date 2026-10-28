import os
from dotenv import load_dotenv
import gradio as gr
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

load_dotenv()

class PDFChatApp:
    def __init__(self,  db_name="pdf_db"):
        self.model = "llama-3.3-70b-versatile"
        self.db_name = db_name
        
        self.documents = []
        self.chunks = []
        self.vectorstore = None
        self.conversation_chain = None
    
    def ensure_temp_dir(self):
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
    
    def upload_pdf(self, pdf_files):
        self.documents = []
        for pdf_file in pdf_files:
            if type(pdf_file) is str:
                file_path = pdf_file
            else:
                file_path = pdf_file.name
            loader = PyPDFLoader(file_path)
            self.documents.extend(self._add_metadata(doc, "pdf") for doc in loader.load())
        
        self.chunks = self.split_documents()
        self.vectorstore = self.create_vectorstore()
        self.conversation_chain = self.create_conversation_chain()
        
        return f"Uploaded {len(pdf_files)} PDF files and processed {len(self.documents)} documents into {len(self.chunks)} chunks."
    
    def _add_metadata(self, doc, doc_type):
        doc.metadata["doc_type"] = doc_type
        return doc
    
    def split_documents(self, chunk_size=1000, chunk_overlap=200):
        if not self.documents:
            return []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(self.documents)
        return chunks
    
    def create_vectorstore(self):
        if not self.chunks:
            return None
        
        embeddings = HuggingFaceEmbeddings()
        if os.path.exists(self.db_name):
            Chroma(persist_directory=self.db_name, embedding_function=embeddings).delete_collection()
        
        vectorstore = Chroma.from_documents(
            documents=self.chunks,
            embedding=embeddings,
            persist_directory=self.db_name
        )
        return vectorstore
    
    def create_conversation_chain(self, k=4):
        if not self.vectorstore:
            return None
            
        llm = ChatGroq(model_name=self.model)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        
        chain_args = {
            "llm": llm,
            "retriever": retriever,
            "memory": memory,
        }
            
        return ConversationalRetrievalChain.from_llm(**chain_args)
    
    def query(self, question):
        if not self.conversation_chain:
            return "Please upload PDF files first."
        result = self.conversation_chain.invoke({"question": question})
        return result["answer"]

    #
    def chat_handler(self, message, history):
        return self.query(message)

    def launch_app(self):
        with gr.Blocks() as app:
            with gr.Row():
                gr.Markdown("# PDF Chat Application")
            
            with gr.Row():
                pdf_files = gr.File(file_count="multiple", label="Upload PDF Files")
                upload_button = gr.Button("Process PDFs")
                status_text = gr.Textbox(label="Status")
            
            upload_button.click(
                fn=self.upload_pdf,
                inputs=[pdf_files],
                outputs=[status_text]
            )
            
            chatbot = gr.ChatInterface(
                fn=self.chat_handler,
                examples=["What's in these documents?", "Summarize the key points"],
                title="Chat with your PDFs"
            )
            
        return app.launch(inbrowser=True)

if __name__ == "__main__":
    app = PDFChatApp()
    app.launch_app()
