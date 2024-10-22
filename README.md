# Команда "улыбка радуги" — определение зарплатных ожиданий по резюме

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![sentence-transformers](https://img.shields.io/badge/sentence--transformers-blue?style=for-the-badge&logo=appveyor)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

## О проекте
Команда "улыбка радуги" представляет две модели для предсказания зарплаты и подходящей профессии для соискателя работы. Помимо этого, нами был разработан алгоритм для группировки похожих профессий, а также предложены два способа исправления опечаток — нейросетевой и аналитический.

Определение профессии кандидата происходит в результате обработки текста резюме с помощью нейросетей: трансформера BERT и перцептрона.
Предсказание зарплаты реализовано с помощью базовых алгоритмов машинного обучения.

Технические особенности и уникальность: 
- алгоритм группировки списка профессий с помощью лемматизации текста, разработанных специально для данной задачи
- вывод новых признаков и зависимостей на базе имеющихся
- предложены подходы на базе GPT для решения задач кейса

## Структура проекта
- `full_launch.ipynb` - ноутбук с прогоном по тестовым данным и формированием сабмита
- `resume_clf_mlp.ipynb` — ноутбук с обработкой датасетов с резюме и обучением классификатора профессий
- `full_launch.ipynb` — ноутбук с полным прогоном по тестовым данным и формированием сабмита
- `process_job_list.ipynb` - ноутбук с обработкой списка профессий и его группировкой
- `inference_regression.py` - файл для инференса регрессионной модели
- `salary_catboost.ipynb` - ноутбук с кодом обучения регрессионной модели
- `SAL_model` - веса регрессионной модели

## Гайд для пользователя
Скачать веса RuBERT-tiny можно здесь: https://www.dropbox.com/scl/fi/rr3pfdga4j3ctfl4ntjok/rubert-tiny-weights.zip?rlkey=jtqcxm12l6xagknbdpyle9rsw&st=87mm93r9&dl=0

Ссылка на базовый MLP: https://www.dropbox.com/scl/fi/rail4zvoq9cdm4dn4j2vy/mlp1.pt?rlkey=j4szpyjd1t3inbma6k88a6cnw&st=b2xzedci&dl=0

