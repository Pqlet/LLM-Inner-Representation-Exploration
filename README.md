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
