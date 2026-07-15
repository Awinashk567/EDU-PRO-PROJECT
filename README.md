🎓 EduPro Predictive Analytics & Revenue Forecaster

📌 Project Overview
EduPro historically relied on intuitive, reactive planning for course launches and pricing, leading to suboptimal resource allocation and 'dead inventory' courses. This project transforms the platform's strategy by introducing a **Machine Learning-driven predictive analytics system**. 

By analyzing historical transaction, faculty, and course data, this project forecasts future course enrollments and revenue, enabling stakeholders to shift from reactive reporting to **proactive, data-driven planning**.

🚀 Key Features
* Dual-Target Forecasting: Predicts both `Enrollment Count` (Volume) and `Course Revenue` (Financial Yield) using a highly accurate Linear Regression model (98% $R^2$ Score).
* Strategic Feature Engineering: Includes the custom `Expertise_Match` metric to objectively align faculty qualifications with course categories, optimizing instructor allocation.
* Dynamic Price Elasticity Simulator: An interactive 'out-of-the-box' feature allowing executives to simulate global price shifts (-50% to +50%) and instantly quantify the resulting impact on total platform volume.
* Interactive Streamlit Dashboard: A user-friendly web interface designed for non-technical stakeholders to test pre-launch parameters and receive real-time business intelligence.

🛠️ Technology Stack
* Language: Python
* Data Manipulation & Analysis: Pandas, NumPy
* Machine Learning: Scikit-Learn (Linear Regression, Random Forest Regressor, StandardScaler, LabelEncoder)
* Data Visualization: Matplotlib, Seaborn
* Deployment & UI: Streamlit

📈 Key Business Insights
1. The Small Sample Paradox: On an aggregated 60-row dataset, simple Linear Regression (98% accuracy) outperformed complex Random Forest models (94% accuracy) which suffered from overfitting.
2. Demand Drivers: Machine learning feature importance analysis proved that Course Duration and Price Sensitivity carry the highest mathematical weight in determining future enrollments.
3. Data Bias Identification: Automated diagnostics uncovered historical biases in instructor assignments, emphasizing the need for data-driven faculty realignment.

💻 How to Run the Project Locally

1. Clone the repository:**
```bash
git clone [https://github.com/Awinashk567/EDU-PRO-PROJECT.git](https://github.com/Awinashk567/EDU-PRO-PROJECT.git)
cd EDU-PRO-PROJECT
2. Install required dependencies:
pip install -r requirements.txt
3. Run the Streamlit Dashboard:
streamlit run app.py
Developed by Awinash Kumar | Data Analyst
