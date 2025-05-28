import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, date # Ensure 'date' is imported
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import numpy as np
import os # For checking file existence

import nltk

# Ensure required NLTK data is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("taggers/averaged_perceptron_tagger")
except LookupError:
    nltk.download("averaged_perceptron_tagger")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")



# --- Configuration ---
DATABASE_FILE = 'tasks.csv'
STOPWORDS = set(stopwords.words('english'))
PROCRASTINATION_KEYWORDS = ["start", "research", "plan", "think about", "figure out", "conceptualize"]

# Define all possible categories for consistent encoding
ALL_EFFORTS = ['Low', 'Medium', 'High']
ALL_TYPES = ['Reading', 'Assignment', 'Revision', 'Project', 'Other']
ALL_USER_PRIORITIES = ['Low', 'Medium', 'High']
ALL_AI_PRIORITIES = ['Low', 'Medium', 'High'] # Possible output priorities for AI

# --- Helper Functions ---

def load_tasks():
    """Loads tasks from CSV or creates an empty DataFrame, ensuring all necessary columns exist."""
    if os.path.exists(DATABASE_FILE):
        try:
            df = pd.read_csv(DATABASE_FILE)
            # Ensure correct data types after loading
            df['Due Date'] = pd.to_datetime(df['Due Date'])

            # Add new columns if they don't exist (for backward compatibility)
            if 'Actual_AI_Priority' not in df.columns:
                df['Actual_AI_Priority'] = None # Or pd.NA
            if 'Actual_Effort_Rating' not in df.columns:
                df['Actual_Effort_Rating'] = None # Or pd.NA
            return df
        except pd.errors.EmptyDataError:
            # Handle case where file exists but is empty
            return pd.DataFrame(columns=['Task Name', 'Course', 'Due Date', 'Effort', 'Type', 'User Priority', 'AI Priority', 'Status', 'Keywords', 'Days Until Due', 'Actual_AI_Priority', 'Actual_Effort_Rating'])
    else:
        return pd.DataFrame(columns=['Task Name', 'Course', 'Due Date', 'Effort', 'Type', 'User Priority', 'AI Priority', 'Status', 'Keywords', 'Days Until Due', 'Actual_AI_Priority', 'Actual_Effort_Rating'])


def save_tasks(df):
    """Saves tasks to CSV."""
    df.to_csv(DATABASE_FILE, index=False)

def calculate_days_until_due(due_date):
    """Calculates days remaining until due date."""
    if isinstance(due_date, pd.Timestamp):
        due_date = due_date.to_pydatetime()
    elif isinstance(due_date, date): # Corrected to use 'date' directly
        due_date = datetime.combine(due_date, datetime.min.time())

    # Calculate days until due, handle already passed dates
    delta = due_date - datetime.now()
    return max(0, delta.days) # Don't show negative days, just 0 for overdue/today

def extract_keywords(text):
    """Extracts keywords from task description (simple NLP)."""
    text = text.lower()
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.isalnum() and word not in STOPWORDS]
    return ", ".join(filtered_words)

def train_priority_model(df_tasks):
    """Trains a simple Logistic Regression model for task prioritization."""
    # Filter for tasks that have actual feedback for training
    df_trainable = df_tasks.dropna(subset=['Actual_AI_Priority', 'Actual_Effort_Rating']).copy()

    if len(df_trainable) < 5: # Need enough data with 'Actual_AI_Priority' to train effectively
        st.sidebar.info("AI model learning from your feedback: Add more tasks and mark them complete with feedback for smarter prioritization! 🧠")
        return None, None, None, None, None

    # Initialize encoders by fitting on all possible categories defined globally
    le_effort = LabelEncoder().fit(ALL_EFFORTS)
    le_type = LabelEncoder().fit(ALL_TYPES)
    le_user_priority = LabelEncoder().fit(ALL_USER_PRIORITIES)
    le_ai_priority = LabelEncoder().fit(ALL_AI_PRIORITIES) # For AI prediction output


    df_encoded = df_trainable.copy()
    
    # Apply encoding. Use .get() or check for class existence to prevent errors if a category isn't in training data
    df_encoded['Effort_encoded'] = df_encoded['Effort'].apply(lambda x: le_effort.transform([x])[0] if x in le_effort.classes_ else -1)
    df_encoded['Type_encoded'] = df_encoded['Type'].apply(lambda x: le_type.transform([x])[0] if x in le_type.classes_ else -1)
    df_encoded['User_Priority_encoded'] = df_encoded['User Priority'].apply(lambda x: le_user_priority.transform([x])[0] if x in le_user_priority.classes_ else -1)
    
    # Filter out rows where encoding might have failed (value was not in le_effort.classes_)
    df_encoded = df_encoded[df_encoded['Effort_encoded'] != -1]
    df_encoded = df_encoded[df_encoded['Type_encoded'] != -1]
    df_encoded = df_encoded[df_encoded['User_Priority_encoded'] != -1]

    if df_encoded.empty or len(df_encoded) < 2:
        st.sidebar.warning("Not enough valid, encoded data with feedback to train AI model. Try again with more varied completed tasks. 🤔")
        return None, None, None, None, None

    # Features for the model
    X = df_encoded[['Days Until Due', 'Effort_encoded', 'Type_encoded', 'User_Priority_encoded']]
    # Target variable: Actual_AI_Priority (this is the key change!)
    y = df_encoded['Actual_AI_Priority']

    # Ensure no NaN or inf in features, fill with mean as a fallback
    X = X.replace([pd.NA, pd.NaT, np.inf, -np.inf], np.nan).fillna(X.mean(numeric_only=True))
    y = y.replace([pd.NA, pd.NaT, np.inf, -np.inf], np.nan).fillna(y.mode()[0]) # Fill target with mode

    # Ensure y has at least 2 unique classes for stratification in train_test_split
    try:
        y_encoded = le_ai_priority.transform(y)
    except ValueError as e:
        st.sidebar.warning(f"Could not encode 'Actual AI Priority' for training: {e}. Ensure feedback uses valid priorities (Low, Medium, High).")
        return None, None, None, None, None

    if len(np.unique(y_encoded)) < 2:
        st.sidebar.warning("Not enough variety in 'Actual AI Priority' labels to train effectively. Provide feedback for tasks with different actual priorities. 📉")
        return None, None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    return model, le_ai_priority, le_effort, le_type, le_user_priority

def predict_priority(model, task_data, le_ai_priority, le_effort, le_type, le_user_priority):
    """Predicts AI Priority for a new task."""
    if model is None:
        return "N/A"

    df_single_task = pd.DataFrame([task_data])

    try:
        # Apply the same encoding. Handle cases where a category might not be known to the encoder.
        df_single_task['Effort_encoded'] = df_single_task['Effort'].apply(lambda x: le_effort.transform([x])[0] if x in le_effort.classes_ else -1)
        df_single_task['Type_encoded'] = df_single_task['Type'].apply(lambda x: le_type.transform([x])[0] if x in le_type.classes_ else -1)
        df_single_task['User_Priority_encoded'] = df_single_task['User Priority'].apply(lambda x: le_user_priority.transform([x])[0] if x in le_user_priority.classes_ else -1)
    except Exception as e:
        st.warning(f"Error during encoding for prediction: {e}. Check if task categories match training categories.")
        return "N/A"

    # If any feature could not be encoded, we cannot predict reliably
    if any(df_single_task[['Effort_encoded', 'Type_encoded', 'User_Priority_encoded']].iloc[0] == -1):
        return "N/A"

    X_predict = df_single_task[['Days Until Due', 'Effort_encoded', 'Type_encoded', 'User_Priority_encoded']]
    
    X_predict = X_predict.fillna(0) # Simple fill for demonstration if a NaN snuck in

    predicted_label = model.predict(X_predict)
    return le_ai_priority.inverse_transform(predicted_label)[0]

def check_for_procrastination(task_name):
    """Simple check for procrastination keywords in task name."""
    task_name_lower = task_name.lower()
    for keyword in PROCRASTINATION_KEYWORDS:
        if re.search(r'\b' + re.escape(keyword) + r'\b', task_name_lower):
            return True
    return False

# --- Streamlit App ---

st.set_page_config(page_title="Smart Study Planner", layout="wide", initial_sidebar_state="expanded")

st.title("📚 Smart Study Planner & Procrastination Nudge")
st.markdown("Your AI-powered assistant to manage tasks and stay on track!")

# Load existing tasks with a spinner
with st.spinner('Loading your tasks...'):
    df_tasks = load_tasks()

# --- Sidebar for Adding Tasks (inside an expander) ---
with st.sidebar:
    st.header("✨ Add New Study Task")
    with st.expander("Click to add a task!", expanded=False):
        with st.form("task_form"):
            task_name = st.text_input("Task Name", help="e.g., Read Chapter 5, Start research for AI project")
            course = st.text_input("Course/Subject", help="e.g., Computer Science, History")
            due_date = st.date_input("Due Date", min_value=datetime.now().date())
            effort = st.selectbox("Estimated Effort", ALL_EFFORTS)
            task_type = st.selectbox("Task Type", ALL_TYPES)
            user_priority = st.selectbox("Your Priority", ALL_USER_PRIORITIES)

            submitted = st.form_submit_button("Add Task")

            if submitted:
                if task_name and course and due_date:
                    days_until_due = calculate_days_until_due(due_date)
                    keywords = extract_keywords(task_name)

                    initial_ai_priority = user_priority # Placeholder, will be updated by model

                    new_task = {
                        'Task Name': task_name,
                        'Course': course,
                        'Due Date': datetime.combine(due_date, datetime.min.time()),
                        'Effort': effort,
                        'Type': task_type,
                        'User Priority': user_priority,
                        'AI Priority': initial_ai_priority,
                        'Status': 'Pending',
                        'Keywords': keywords,
                        'Days Until Due': days_until_due,
                        'Actual_AI_Priority': None, # New column
                        'Actual_Effort_Rating': None # New column
                    }
                    df_tasks = pd.concat([df_tasks, pd.DataFrame([new_task])], ignore_index=True)
                    save_tasks(df_tasks)
                    st.success("Task added successfully! 🎉 Your planner is updated.")
                    st.rerun()
                else:
                    st.error("Please fill in all task details!")

# --- Main Content Area ---

st.subheader("📋 My Study Tasks")

if df_tasks.empty:
    st.info("No tasks added yet. Click '✨ Add New Study Task' in the sidebar to get started! 🚀")
else:
    # Retrain model periodically or when new tasks are added
    # Initialize these outside the if model: block to ensure they are always defined
    model, le_ai_priority, le_effort, le_type, le_user_priority = None, None, None, None, None 
    
    with st.spinner('AI learning from your feedback...'):
        # Only train if there are enough tasks with actual feedback
        if len(df_tasks.dropna(subset=['Actual_AI_Priority', 'Actual_Effort_Rating'])) >= 5:
            model, le_ai_priority, le_effort, le_type, le_user_priority = train_priority_model(df_tasks)
        elif len(df_tasks) >= 5: # If there are enough tasks but not enough feedback yet
             st.info("💡 Once you mark 5 or more tasks complete and provide feedback, the AI will start learning your personal prioritization style!")

    # Update AI Priority for all *pending* tasks using the trained model if model is available
    if model:
        # Create a copy to avoid SettingWithCopyWarning
        df_tasks_copy = df_tasks.copy() 
        for index, row in df_tasks_copy.iterrows():
            if row['Status'] == 'Pending': # Only update AI priority for pending tasks
                task_data_for_prediction = {
                    'Days Until Due': row['Days Until Due'],
                    'Effort': row['Effort'],
                    'Type': row['Type'],
                    'User Priority': row['User Priority']
                }
                predicted_prio = predict_priority(model, task_data_for_prediction, le_ai_priority, le_effort, le_type, le_user_priority)
                df_tasks_copy.loc[index, 'AI Priority'] = predicted_prio
        df_tasks = df_tasks_copy # Update the main DataFrame
        save_tasks(df_tasks) # Save updated AI priorities


    # Sort tasks by AI Priority (High > Medium > Low) and then by Due Date
    priority_order = {'High': 2, 'Medium': 1, 'Low': 0, 'N/A': -1}
    df_tasks['AI Priority Value'] = df_tasks['AI Priority'].map(priority_order)
    # Sort by status (Pending first), then AI priority, then due date
    df_tasks_sorted = df_tasks.sort_values(by=['Status', 'AI Priority Value', 'Due Date'], ascending=[True, False, True]).drop(columns=['AI Priority Value']) 

    # Display DataFrame with enhanced styling
    st.dataframe(
        df_tasks_sorted[['Task Name', 'Course', 'Due Date', 'Effort', 'Type', 'User Priority', 'AI Priority', 'Status']].style.applymap(
            lambda x: 'background-color: #ffe0b2' if x == 'High' else ('background-color: #fff3e0' if x == 'Medium' else ''),
            subset=['AI Priority']
        ).applymap(
            lambda x: 'text-decoration: line-through; color: gray;' if x == 'Completed' else '',
            subset=['Status']
        ).hide(axis="index"), # Hide the default pandas index
        use_container_width=True,
        height=300 # Adjust height for better display
    )

st.markdown("---")

st.subheader("💡 Procrastination Nudge")
st.markdown("Here are some tasks that might need a bit more focus, based on their description and priority:")

procrastination_tasks = []
if not df_tasks.empty:
    pending_tasks = df_tasks[df_tasks['Status'] == 'Pending']
    for index, task in pending_tasks.iterrows():
        if task['AI Priority'] != 'N/A' and check_for_procrastination(task['Task Name']) and task['AI Priority'] in ['High', 'Medium']:
            procrastination_tasks.append(task['Task Name'])

    if procrastination_tasks:
        for task_name in procrastination_tasks:
            st.warning(f"**{task_name}** – This task seems like a 'start' or 'plan' task. Break it down or set a small, actionable goal! 🐢")
    else:
        st.info("No immediate signs of procrastination detected among your high-priority tasks. You're crushing it! 🎉")
else:
    st.info("Add tasks to see procrastination nudges here!")

st.markdown("---")

st.subheader("Action Center")
# Use columns for action buttons
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### ✅ Mark Task as Complete")
    # Filter for pending tasks that have not yet provided feedback
    tasks_to_complete_feedback = df_tasks[(df_tasks['Status'] == 'Pending')].copy()
    
    if not tasks_to_complete_feedback.empty:
        # Use a unique key for the selectbox to avoid conflicts if used elsewhere
        selected_task_name_complete = st.selectbox("Which task did you finish?", tasks_to_complete_feedback['Task Name'].tolist(), key="complete_task_select")
        
        # Get the current task details for the feedback form
        current_task_row = df_tasks[df_tasks['Task Name'] == selected_task_name_complete].iloc[0]

        with st.form("feedback_form"):
            st.markdown(f"**Feedback for: '{selected_task_name_complete}'**")
            actual_priority_feedback = st.selectbox("What was its *actual* importance/priority?", ALL_AI_PRIORITIES)
            actual_effort_feedback = st.selectbox("How was the *actual* effort compared to your estimate?", ['Shorter', 'As Estimated', 'Longer'])
            
            feedback_submitted = st.form_submit_button("Submit Feedback & Complete Task ✅")

            if feedback_submitted:
                with st.spinner(f"Completing '{selected_task_name_complete}' and saving feedback..."):
                    idx_to_update = df_tasks[df_tasks['Task Name'] == selected_task_name_complete].index[0]
                    df_tasks.loc[idx_to_update, 'Status'] = 'Completed'
                    df_tasks.loc[idx_to_update, 'Actual_AI_Priority'] = actual_priority_feedback
                    df_tasks.loc[idx_to_update, 'Actual_Effort_Rating'] = actual_effort_feedback
                    save_tasks(df_tasks)
                    st.success(f"'{selected_task_name_complete}' marked as complete with feedback! Great job! 🥳")
                    st.rerun()
    else:
        st.info("No pending tasks to mark complete or provide feedback for! ✨")

with col2:
    st.markdown("#### 🗑️ Delete Task")
    tasks_to_delete = df_tasks['Task Name'].tolist()
    if tasks_to_delete:
        selected_task_to_delete = st.selectbox("Which task do you want to remove?", tasks_to_delete, key="delete_task")
        if st.button("Delete Selected Task 🗑️"):
            with st.spinner(f"Deleting '{selected_task_to_delete}'..."):
                df_tasks = df_tasks[df_tasks['Task Name'] != selected_task_to_delete].reset_index(drop=True)
                save_tasks(df_tasks)
                st.success(f"'{selected_task_to_delete}' deleted successfully!")
                st.rerun()
    else:
        st.info("No tasks to delete.")

st.markdown("---")
st.caption("Built with ❤️ and AI for productive studying.")
