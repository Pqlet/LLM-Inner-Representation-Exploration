# LLM-Inner-Representation-Exploration

## Выводы

- Эмбеддинги текстов в разных письменностях имеют разные распределения, что показывает ID, KL дивергенция и анизотропия 

- Связь внутренней размерности с морфологической сложностью языков и малоресурсностью не прослеживается

- Неграмматичные тексты имеют бóльшую внутреннюю размерность

- Внутренняя размерность представлений линейно связана с качеством модели

## Ноутбуки

- `languages-exps` - эксперименты с языками, неграмматичным текстами и BPC (bits per character)
- `model-runs` - эксперименты с разными моделями, подсчёт Intrinsic Dimensionality и Anisotropy
- `results` - сохраненные метрики
- `language samples` - таблицы с выборками языков
- `internal_representations.pdf` - отчет

## Эксперименты

Наша работа посвящена анализу геометрических характеристик эмбеддингов языковых моделей для разных языков. Мы сравнивали внутреннюю размерность, анизотропию и KL дивергенцию между различными языками и английским. **Более подробно в нашем отчете в PDF файле.**

Внутренняя размерность неграмматичных текстов выше.
<img src="https://github.com/Pqlet/LLM-Inner-Representation-Exploration/blob/main/imgs/id_ungrammatical.png" height="300"/>

ID эмбеддингов языков линейно связана с качеством модели. Чем меньше ID, тем выше качество на MMLU.
<img src="https://github.com/Pqlet/LLM-Inner-Representation-Exploration/blob/main/imgs/5368378111621717603.jpg" height="400"/>

Для языков, использующих две письменности:
KL дивергенция с английским выше, если запись латиницей
<img src="https://github.com/Pqlet/LLM-Inner-Representation-Exploration/blob/main/imgs/kl_no_agg_alph.jpg" />

Анизотропия обученного декодера имеет характерную форму для всех языков.
<img src="https://github.com/Pqlet/LLM-Inner-Representation-Exploration/blob/main/imgs/anisotropy_agg_no_rand.jpg" />


## Оптимизация метода оценки анизотропии с помощью степенного метода
Оценка анизотропии пространства эмбедингов может быть сделана с помощью Cosine Similarity или Maximum Explained Variance (MEV), как в работе Ethayarajh 2019. Мы используем второй способ, аналогично Razzhigaev et al. 2023: 
```math
\text{anisotropy}(X) = \frac{\sigma^2_1}{\sum^k_{i=1}\sigma^2_i}
```
, но вычисляем первое сингулярное значение с помощью Степенного метода, а знаменатель по формуле Фробениусовой нормы матрицы эмбедингов. Данная имплементация подсчёта уменьшила временную сложность вычисления анизотропии, потому что не нужно считать полное SVD разложение.

Имплементация на Python:
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
