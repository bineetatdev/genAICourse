import numpy as np
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class VizEmbedding:
    def __init__(self):
        self.tokenizer = self.get_tokenizer()
        self.model = self.get_model()
    
    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
    
    def get_model(self):
        return AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
    
    def get_embeddings(self, document):
        inputs = self.tokenizer(document, return_tensors='pt')
        token_ids = inputs['input_ids'][0]
        model_output = self.model(**inputs)
        embeddings = model_output[0]
        return token_ids, embeddings

    def find_token_indices(self, token_ids, target_token_id):
        return [i for i, token_id in enumerate(token_ids) if token_id == target_token_id]
    
    def get_token_embedding(self, embeddings, token_index):
        return embeddings[0, token_index, :].detach().numpy()
    

if __name__ == "__main__":
    
    v = VizEmbedding()
    
    sentences = [
        "I need to visit the bank to deposit my paycheck before it closes at 5 PM.",
        "My grandmother still doesn't trust online banking and insists on going to the bank in person.",
        "The bank charged me an overdraft fee when my account went below zero.",
        "Several branches of the bank closed during the financial crisis of 2008.",
        "The investment bank helped the company go public through an IPO.",
        "I'm applying for a loan at the bank to finance my new home.",
        "The central bank announced a change in interest rates yesterday.",
        "She works as a teller at the bank on Main Street.",
        "The bank vault contains millions in cash and valuable assets.",
        "After the robbery, the bank installed additional security cameras.",
        
        "We had a picnic on the grassy bank overlooking the Mississippi River.",
        "Erosion has significantly damaged the river bank over the last decade.",
        "The fisherman sat patiently on the bank, waiting for a bite.",
        "Several turtles were sunning themselves on the muddy bank.",
        "Willow trees grow densely along the eastern bank of the creek.",
        "The kayakers pulled their boats onto the bank for lunch.",
        "Spring floods often cause the river bank to overflow.",
        "The archaeologists found ancient artifacts buried in the west bank of the river.",
        "We built our cabin just fifty yards from the lake's bank.",
        "The otter family made their home in a burrow dug into the bank.",
        
        "I bank on you for such theoretical content",
        "The company will bank on innovation to stay ahead of competitors.",
        "You can bank on her to complete the project by the deadline.",
        "The coach will bank on the team's strong defense to win championships.",
        "Let us bank on the weather forecast being accurate this time of year.",
        "The campaign will bank on young voters turning out in record numbers.",
        "I will bank on that investment providing immediate returns.",
        "She will bank on her years of experience when facing new challenges.",
        "We're will bank on your support to get this legislation passed.",
        "The school does bank on alumni donations to fund scholarship programs."
    ]
    bank_embeddings_list = []

    bank_token_info = v.tokenizer("bank", add_special_tokens=False)
    bank_token_id = bank_token_info['input_ids'][0]

    for sentence in sentences:
        tokens, embeddings = v.get_embeddings(sentence)
        bank_indices = v.find_token_indices(tokens, bank_token_id)
        bank_embedding = v.get_token_embedding(embeddings, bank_indices[0])

        bank_embeddings_list.append(bank_embedding)
    
    bank_embeddings_matrix = np.array(bank_embeddings_list)

    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    reduced_embeddings = tsne.fit_transform(bank_embeddings_matrix)

    colors = ['red'] * 10 + ['blue'] * 10 + ['green'] * 10

    plt.figure(figsize=(8, 6))
    for i in range(30):
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], color=colors[i])
        plt.text(reduced_embeddings[i, 0] + 0.5, reduced_embeddings[i, 1] + 0.5, str(i), fontsize=9)
        
    plt.title("t-SNE of Bank Embeddings (3 color-coded groups)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.show()

    
    