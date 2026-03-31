## Comparative Analysis between Machine Learning and Neural Networks for heatwave prediction system

### Project Overview
Heatwaves are extreme weather events that endanger human health, agriculture, and energy supply and demand. It is important to forecast the risk of heatwave in advance for the planning and the disaster management.
In this work we developed a heatwave forecasting system based on both ML models and a FFNN. It categorizes the risk of a heatwave as either Low, Medium or High by using the real-world weather information, and also explains predictions with eXplainable AI methodologies. 

### Objectives
- Collect and preprocess real-world weather data
- Analyze relationships between weather factors and heatwaves
- Build machine learning models for prediction
- Develop a neural network using the same dataset
- Compare performance of ML vs Neural Network
- Identify important features influencing heatwaves
- Classify heatwave risk into Low, Medium, High
- Provide insights for early warning systems

### Dataset Information
- Source: Visual Crossing Weather Data
- Location: Chennai, India
- Time Period: 2010 – 2023
- Total Records: 5932
- Total Features: 20

Key Features include temperature, humidity, wind speed, precipitation metrics, solar radiation, UV index, pressure, and geographical coordinates.

### Methodology
Data Collection → Data Preprocessing → Feature Selection → Model Training → Model Evaluation → Risk Classification

Steps:
1. Data cleaning and preprocessing
2. Feature selection and scaling
3. Training Machine Learning models
4. Training Feedforward Neural Network
5. Model evaluation and comparison
6. Explainability using feature importance techniques

### Models Used
Machine Learning:
- Random Forest 
- Decision Tree
- Support Vector Machine
- Logistic Regression

Neural Network:
- Feedforward Neural Network

eXplainable AI:
- LIME

### Results and Evaluation
Models are evaluated based on classification performance, focusing on accuracy, generalization, and interpretability. Machine Learning models provide stable and interpretable results, while Neural Networks capture more complex patterns.

### Comparison
Aspect              Machine Learning      Neural Network
Interpretability        High                    Low
Complexity              Low                     High
Training Time           Fast                    Slower
Pattern Learning        Moderate                High

### Explainability and Insights
Heatwaves are influenced by multiple interacting weather factors. Temperature alone is insufficient for accurate prediction. Key contributing features include temperature, humidity, solar radiation, and wind speed.

### Applications
- Heatwave early warning systems
- Disaster management planning
- Public safety alerts
- Environmental monitoring

### Limitations
- Dataset limited to Chennai region
- Moderate dataset size
- Neural network requires tuning
- Weather variability may affect predictions

### Future Work
- Extend to multi-city datasets
- Use time-series models such as LSTM
- Develop real-time prediction systems
- Deploy as a web-based application

### Technology
- Python
- scikit-learn
- pandas, numpy
- matplotlib, seaborn

### Project Folder Structure
data/
models/
notebooks/
src/
README.md

### Installation and Usage
pip install -r requirements.txt
python main.py

### Reference
Comparative analysis of machine learning approaches for heatwave event prediction in India Ritesh Choudary V1, Anita Christaline Johnvictor2 & Prem Sankar N3

### Author
Shyma Sundaram S P
23MID0341
M.Tech Integrated Computer Science and Engineering (Data Science)
Vellore Institute of Technology
