# Great things are coming
This project was initially designed to deploy a streamlit frontend PoC for a custom LangChain orchestrated RAG system. Due to various limitations, it has been converted into an ongoing experimentation lab for LLM orchestration libraries.

The following use-cases have been identified (as of now): 
1. LangChain orchestrated RAG system deployed with StreamLit frontend.  
2. LlamaIndex orchestrated RAG systems deployed with ChainLit frontend.
3. Evaluation pipeline for baseline RAG system with LlamaIndex.
4. Evaluation of document retrieval methodologies (HyDE and sub-question query decomposition).

![Image](images/rag_abstraction.png)

---

## **Setup Instructions**  

Follow these steps in the specified order to run the scripts successfully:

### **1. Clone the Repository**  
```bash
git clone https://github.com/jinkett99/great-things.git
cd great-things
```

### **2. Install Dependencies**  
```bash
pip install -r requirements.txt
```

---

## **Content**
```
RAG-webscraper/
notebooks/
├── 0.01-jk-experiments.ipynb           # Write runnable .python files for streamlit, play with various aspects of RAG chain (Index, Document retrieval, Query engine)
```

```
RAG-evaluation/
notebooks/
├── 0.01-jk-evaluation_pipeline.ipynb   # Build a custom RAG system (query engine) with LlamaIndex, run evaluation scripts
├── 0.01-jk-query_rewriting.ipynb       # Implementation and experimentation of HyDE and Sub Question Query Engines (Document retrieval methodologies) + Evaluation
```

```
main/
├── app.py                              # Spins up ChainLit frontend at http://localhost:8000, run command "chainlit run app.py -w"
├── 0.01-jk-query_rewriting.ipynb       # Implementation and experimentation of HyDE and Sub Question Query Engines (Document retrieval methodologies) + Evaluation
```

---

## **Contributing**  
Feel free to open issues and submit pull requests. Contributions are welcome!
