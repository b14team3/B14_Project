# B14_Project
# **Advanced Skin Disease Diagnosis Leveraging Machine Learning**

## **Overview**
The **Advanced Skin Disease Diagnosis** system combines machine learning and image processing to provide accurate and insightful analysis of skin conditions. By integrating secure user functionalities, robust image processing capabilities, and a voting ensemble model, this solution empowers users with real-time diagnostic capabilities directly from uploaded images. This tool aims to revolutionize medical diagnosis accessibility and efficiency, focusing on skin disease identification and analysis.

---

## **Use Case**
The project targets individuals seeking initial insights into skin conditions before consulting a dermatologist. By leveraging machine learning and advanced image processing techniques, the system aids in early diagnosis, thereby potentially reducing the time and cost associated with medical consultations. It is especially valuable in areas with limited access to dermatological expertise.

---

## **Milestones**

### **Milestone 1: Development of Customer Features**
1. **Create Account and Login Functionalities**
   - **User Registration**: Secure account creation with fields for username, email, and password.
   - **Login**: Authentication with secure credentials for personalized user access.
   - **Session Management**: Persistent sessions for seamless user experience.

2. **Image Upload Feature**
   - **User Submission**: Allows users to upload skin condition images for analysis.
   - **Image Validation**: Ensures valid image formats (JPEG, PNG) and appropriate file sizes.

3. **View Page for Results**
   - **Diagnostic Display**: Provides clear and insightful analysis results based on the uploaded image.

---

### **Milestone 2: Mastering Image Processing with OpenCV**
1. **Foundational and Advanced OpenCV Skills**
   - Comprehensive understanding of OpenCV for image processing.
   - Focused application of techniques tailored to medical image analysis.

2. **Image Pre-Processing and Filtering**
   - **Quality Enhancement**: Noise reduction, histogram equalization, and smoothing.
   - **Contrast Adjustment**: Improved visibility of skin lesions for diagnostic accuracy.

3. **Contour Detection and Segmentation**
   - **Region Isolation**: Contour detection for identifying skin lesions or abnormalities.
   - **Segmentation**: Isolating the affected area for precise analysis.

---

### **Milestone 3: UI Integration of Image Processing**
- Integrated OpenCV into the project to enable real-time image analysis.
- **Key Features**:
  - Users can upload skin condition images.
  - The backend processes images using OpenCV and the Canny Algorithm to detect and measure lesion dimensions.
  - Dimensional data is displayed on the UI, offering users insights into the extent of the condition.

---

### **Milestone 4: Integration of Voting Ensemble Model and Final Implementation**
1. **Model Training and Deployment**
   - **Voting Ensemble Model**: Combines Random Forest, SVM, and Logistic Regression, achieving 80% accuracy in skin condition classification.
   - **Backend Integration**: Links the model with OpenCV for feature extraction and classification.

2. **Real-Time Functionality**
   - Users upload images via the UI.
   - The system processes images, extracts features, classifies the condition, and displays results.
   
3. **Final Outcome**
   - A holistic diagnostic tool that integrates advanced machine learning and image processing for accurate and user-friendly skin condition analysis.

---

## **OUTCOME**

- **Home Page**: _A welcoming interface showcasing the purpose and features of the tool._  
  ![Home Page](path/to/homepage.png)

- **Registration Page**: _Secure user registration form for creating accounts._  
  ![Registration Page](path/to/registration.png)

- **Login Page**: _User-friendly login interface for secure access._  
  ![Login Page](path/to/login.png)

- **Profile Page**: _Displays user information and uploaded images._  
  ![Profile Page](path/to/profile.png)

- **Diagnosis Page**: _Interactive UI displaying diagnostic results and dimensional insights of the condition._  
  ![Diagnosis Page](path/to/diagnosis.png)

---

## **Technologies Used**
1. **Programming Languages**: Python
2. **Frameworks**: Django for backend, OpenCV for image processing
3. **Machine Learning**: Voting Ensemble Model (Random Forest, SVM, Logistic Regression)
4. **Database**: SQLite3
5. **Frontend**: HTML, CSS, JavaScript for UI design

---

## **How to Run**
1. Clone the repository:
   ```bash
   git clone https://github.com/b14team3/B14_Project.git
