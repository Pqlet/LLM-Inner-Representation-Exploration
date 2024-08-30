# LLM-Inner-Representation-Exploration

Наша работа посвящена анализу геометрических характеристик эмбеддингов языковых моделей для разных языков. Мы сравнивали внутреннюю размерность, анизотропию и KL дивергенцию между различными языками и английским. Более подробно в нашем отчете.

Внутренняя размерность неграмматичных текстов выше.
<img src="https://github.com/Pqlet/LLM-Inner-Representation-Exploration/edit/main/imgs/.png" />
ID эмбеддингов языков линейно связана с качеством модели. Чем меньше ID, тем выше качество на MMLU.
<img src="https://github.com/Pqlet/LLM-Inner-Representation-Exploration/edit/main/imgs/mmlu.png" />
Для языков, использующих две письменности:
KL дивергенция с английским выше, если запись латиницей
<img src="https://github.com/Pqlet/LLM-Inner-Representation-Exploration/edit/main/imgs/.png" />
Анизотропия обученного декодера имеет характерную форму для всех языков.
<img src="https://github.com/Pqlet/LLM-Inner-Representation-Exploration/edit/main/imgs/.png" />


## Implementation of the Anisotropy computation with the Power Method
The evaluation of Anisotropy can be done with Cosine Similarity or Maximum Explained Variance (MEV) akin to Ethayarajh 2019. We employ the second approach like in Razzhigaev et al. 2023, but calculate the first singular value using the Power Method and the denominator as the Frobenius norm of the matrix.

The implementation of Anisotropy metric in python:
```python
def calculate_anisotropy_torch(emb):
    embeddings = emb - emb.mean(dim=0, keepdim=True)
    num_iters = 100    
    x = torch.randn(embeddings.shape[0], device=embeddings.device, dtype=emb.dtype)
    for i in range(num_iters):       
        x /= torch.norm(x)
        x = x @ embeddings
        x /= torch.norm(x)
        x = embeddings @ x 
    sigma = torch.norm(x)
    anisotropy = sigma ** 2 / torch.norm(embeddings) ** 2    
    return anisotropy
```
