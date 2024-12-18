# Machine Learning - Projet Cours

Ce projet est une introduction pratique au machine learning, utilisant Python 3.12.8, PyTorch, TensorFlow, CUDA 1.24, et l'environnement Jupyter. Vous pouvez cloner ce dépôt, configurer votre environnement et explorer les exemples et exercices proposés.

---

## 📋 Instructions pour l'installation

### 1. Cloner le projet

Pour commencer, ouvrez un terminal et exécutez la commande suivante :

```bash
git clone https://github.com/skayne11/Machine_learning_cour 
cd Machine_learning_cour`
```


---

### 2. Créer un environnement virtuel Python

Créez un environnement virtuel pour isoler vos dépendances Python :

```bash
python -m venv env
```

Activez l'environnement virtuel :

- Sur **Windows** :

```bash
env\Scripts\activate
```

    
- Sur **Linux/macOS** :

```bash
source env/bin/activate
```

---

### 3. Installer les dépendances

Une fois l'environnement activé, installez les dépendances nécessaires avec la commande :

```bash
pip install -r requirements.txt
```

---

## 🛠 Configuration requise

### Extensions nécessaires

- **Jupyter pour VS Code** : Assurez-vous d'avoir installé l'extension [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) pour exécuter les notebooks directement dans VS Code.
- **Extension PDF** : Installez une extension compatible avec les fichiers PDF dans VS Code, comme [vscode-pdf](https://marketplace.visualstudio.com/items?itemName=tomoki1207.pdf).

---

### CUDA pour PyTorch et TensorFlow

Pour exploiter les performances de votre GPU avec CUDA :

1. Vérifiez que votre carte graphique et vos pilotes NVIDIA prennent en charge CUDA 1.24.
2. Installez les versions compatibles de PyTorch et TensorFlow avec CUDA activé.
3. Consultez la documentation officielle pour des configurations spécifiques :
    - PyTorch CUDA Installation
    - TensorFlow GPU Support

---

## 📄 Notes complémentaires

- **Structure du projet** : Le dépôt contient des notebooks Jupyter avec des exemples et des exercices pratiques. Les données nécessaires sont incluses ou générées directement.

---

## 📚 Ressources utiles

- Documentation PyTorch
- Documentation TensorFlow
- Tutoriel Jupyter Notebook
- [Guide de VS Code pour Python](https://code.visualstudio.com/docs/python/python-tutorial)

---

## 🖥 Configuration système

- **Python** : 3.12.8
- **CUDA** : 1.24
- **Frameworks** : PyTorch, TensorFlow
