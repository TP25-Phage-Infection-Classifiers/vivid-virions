## Entwicklung von Klassifikatoren des maschinellen Lernens für Phagen-Infektionen

> Stand: 07.04.2025

### Hintergrund

In diesem Teamprojekt entwickeln Sie und Ihr Team maschinelle Lernverfahren zur Vorhersage von Phagengenklassen. Phagen sind bakterielle Viren, die ihre bakteriellen Wirte in kurzer Zeit infizieren und abtöten. Daher sind Phagen aufkommende Alternativen zu Antibiotika, um bakterielle Infektionen durch die sogenannte Phagentherapie zu behandeln. Eine Infektion ist ein fein abgestimmtes Genexpressionsprogramm, bei dem bestimmte Phagengene in frühen, mittleren und späten Phasen exprimiert werden, die mit unterschiedlichen Genfunktionen verbunden sind. Diese Genklassen können experimentell durch RNA-Sequenzierung bestimmt werden, was kostspielig und zeitaufwendig ist. Ideal wäre es, wenn man die Genklassen der Phagen direkt aus ihrer Sequenz vorhersagen könnte. Dies ist die Motivation für dieses Teamprojekt, in dem Sie Modelle für maschinelles Lernen (ML) in Python entwickeln werden, um Phagengenklassen anhand ihrer Sequenz vorherzusagen. Für diesen Teil wird Python als Programmiersprache verwendet. Außerdem werden Sie eine leichtgewichtige Benutzerschnittstelle zur Verwendung Ihrer ML-Tools erstellen. Der gesamte Prozess der Software-Entwicklung im Team wird mit SCRUM durchgeführt - ein Rahmenwerk, das in der Software-Entwicklung in der Industrie häufig verwendet wird. Dieses Teamprojekt wird einen Beitrag zu dem aufstrebenden Gebiet der Phagenforschung an der Schnittstelle von Bioinformatik und ML leisten.

![alt text](./.media/tp25_aoc.png)

### Lernziele

In diesem Teamprojekt werden Sie Ihr Wissen über Python auf ML und Softwareentwicklung erweitern. Außerdem werden Sie wesentliche Soft Skills in Team- und Projektarbeit mittels SCRUM sowie Datenpräsentation und Forschung erwerben. Wir werden Ihnen voranalysierte RNA-seq-Daten zur Verfügung stellen, um Ihre Reise zur Vorhersage von Phagengenklassen aus deren Sequenz zu beginnen.

### Ressourcen/Tutorials

- [Interaktives Tutorial zu `scikit-learn`](https://inria.github.io/scikit-learn-mooc/toc.html).
- Einführungen in [Git](https://github.com/git-guides), [GitHub](https://skills.github.com/) und [GitHub Projects](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/quickstart-for-projects).

### Projektstruktur Vorschlag

Dieses Repository kann zudem als Vorlage für eine vorgeschlagene Projektstruktur, welche die Entwicklung und den Einsatz eines maschinellen Lernmodells mit scikit-learn erleichtert, verwendet werden:

- `input/`: Contains dataset files.
- `models/`: Stores trained models in pickle format.
- `notebooks/`: Includes Jupyter notebooks for exploratory data analysis.
- `src/`: Contains scripts for training, prediction, model selection, hyperparameter tuning, and utility functions.
- `requirements.txt`: Lists project dependencies.
- `README.md`: Provides an overview and setup instructions.
