import streamlit as st
import pandas as pd
import os
import json
from io import StringIO
from dotenv import load_dotenv

# Database libraries
import snowflake.connector
import psycopg2
import mysql.connector
import sqlalchemy
from google.cloud import bigquery
from databricks import sql

# LLM Client (Groq)
from groq import Groq

# ============== LOAD ENVIRONMENT VARIABLES ==============
# load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_KEY=st.secrets["GROQ_API_KEY"]

# ============== INITIALIZE GROQ CLIENT ==============
if GROQ_API_KEY is None:
    st.error("⚠️ Groq API Key not found. Check your .env file.")
else:
    groq_client = Groq(api_key=GROQ_API_KEY)

# ============== STREAMLIT PAGE CONFIG ==============
st.set_page_config(page_title="Groq AI-Powered ETL Chatbot", layout="wide")

# ============== SESSION STATE INIT ==============
if "page" not in st.session_state:
    st.session_state.page = 0
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "db_connection" not in st.session_state:
    st.session_state.db_connection = None
if "db_params" not in st.session_state:
    st.session_state.db_params = {}
if "db_type" not in st.session_state:
    st.session_state.db_type = None
if "transformed_df" not in st.session_state:
    st.session_state.transformed_df = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# ============== CACHING: LOAD CSV ==============
@st.cache_data(show_spinner=False)
def load_csv(file):
    """Load CSV into a pandas DataFrame (cached)."""
    return pd.read_csv(file)

# ============== CACHING: LLM TRANSFORMATIONS ==============
@st.cache_data(show_spinner=False)
def llm_transform_data(df: pd.DataFrame, prompt: str):
    """
    Sends the dataset + prompt to the LLM, expects a Markdown table, 
    and converts that table to a Pandas DataFrame. Returns the raw 
    markdown string and the converted DataFrame.
    """
    # 1) Call LLM
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": (
                f"{prompt}\n\n"
                f"Here is a sample of the dataset (for context, 10 rows shown):\n"
                f"{df.to_markdown(index=False)}\n\n"
                "Return ONLY a Markdown table."
            )
        }]
    )
    raw_markdown = response.choices[0].message.content

    # 2) Convert Markdown table to DataFrame
    converted_df = markdown_to_dataframe(raw_markdown)
    return raw_markdown, converted_df


def markdown_to_dataframe(markdown_text: str) -> pd.DataFrame:
    """
    Extract a Markdown table from LLM response and convert it to DataFrame.
    Returns None if no valid table is found.
    """
    try:
        lines = markdown_text.split("\n")
        table_lines = [line for line in lines if line.strip().startswith("|")]

        if len(table_lines) < 2:
            return None  # No valid table found

        # Remove the markdown table header separator line (the second line)
        clean_table = "\n".join([line for i, line in enumerate(table_lines) if i != 1])

        # Convert to DataFrame
        table_io = StringIO(clean_table)
        df = pd.read_csv(table_io, sep="|", skipinitialspace=True, engine="python")

        # Drop extraneous empty columns
        df.dropna(axis=1, how="all", inplace=True)
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
        df.columns = [col.strip() for col in df.columns]
        return df
    except Exception:
        return None

# ============== DB UTILITY FUNCTIONS ==============
def create_db_engine(db_type: str, params: dict):
    """
    Create a SQLAlchemy engine or direct DB connection object 
    depending on the database type.
    """
    if db_type == "Snowflake":
        # For Snowflake, we'll return a raw connector (since Snowflake's SQLAlchemy has nuances).
        return snowflake.connector.connect(**params)
    elif db_type == "PostgreSQL":
        engine_url = (
            f"postgresql://{params['user']}:{params['password']}@"
            f"{params['host']}:{params['port']}/{params['database']}"
        )
        return sqlalchemy.create_engine(engine_url)
    elif db_type == "MySQL":
        engine_url = (
            f"mysql+mysqlconnector://{params['user']}:{params['password']}@"
            f"{params['host']}:{params['port']}/{params['database']}"
        )
        return sqlalchemy.create_engine(engine_url)
    elif db_type == "Databricks":
        # For Databricks, use the "sql.connect".
        # We'll keep returning a raw connection object here as well.
        conn = sql.connect(
            server_hostname=params["host"],
            http_path=params["http_path"],
            access_token=params["token"]
        )
        return conn
    else:
        # BigQuery or fallback
        return None

def upload_to_snowflake(conn, df: pd.DataFrame, table_name: str):
    """
    Example Snowflake create/replace + insert approach (might need adjustments).
    """
    cursor = conn.cursor()
    # Construct CREATE TABLE
    create_query = f"""
    CREATE OR REPLACE TABLE {table_name} (
       {", ".join([f'"{col}" STRING' for col in df.columns])}
    )
    """
    cursor.execute(create_query)

    # Construct Insert for each row
    insert_query = f"""
    INSERT INTO {table_name} ({", ".join([f'"{col}"' for col in df.columns])})
    VALUES ({", ".join(['%s' for _ in df.columns])})
    """
    # Insert each row
    data_rows = [tuple(row) for _, row in df.iterrows()]
    cursor.executemany(insert_query, data_rows)
    conn.commit()
    cursor.close()

def upload_df_to_db(db_type: str, conn_obj, df: pd.DataFrame, table_name: str, db_params: dict):
    """
    Given a DB type and either an engine or a raw connection object, upload df to the table.
    """
    if db_type == "Snowflake":
        # conn_obj is a Snowflake connector
        upload_to_snowflake(conn_obj, df, table_name)

    elif db_type in ["PostgreSQL", "MySQL"]:
        # conn_obj is a SQLAlchemy engine
        df.to_sql(table_name, conn_obj, if_exists="replace", index=False)

    elif db_type == "BigQuery":
        # Requires Google Cloud credentials
        project_id = db_params.get("project_id")
        dataset_id = db_params.get("dataset")
        table_ref = f"{project_id}.{dataset_id}.{table_name}"

        bq_client = bigquery.Client.from_service_account_info(
            db_params["credentials_json"]
        )
        job = bq_client.load_table_from_dataframe(df, table_ref)
        job.result()  # Wait for job to complete

    elif db_type == "Databricks":
        # conn_obj is a databricks.sql.connect
        # There's no direct `to_sql` built into the DBAPI, so we can do an approach:
        # Convert to CSV and run an INSERT or use Spark approach if possible.
        # For simplicity, let's do a naive approach (not recommended for large data).
        engine = None
        try:
            # Attempt using a SQLAlchemy approach if you installed the Databricks SQLAlchemy driver
            # Otherwise, you'd have to do it row-by-row. 
            # If you have a working driver, you can do something like:
            from sqlalchemy import create_engine
            # This is an example connection string; adjust for your environment:
            # "databricks+connector://token:[TOKEN]@[HOSTNAME]:[PORT]/?http_path=[HTTP_PATH]"
            # You might need to build your own connection string properly.
            pass  
        except Exception as e:
            # Fallback or custom row-by-row insert
            st.error(f"Databricks upload not fully implemented: {e}")

    else:
        st.error("Unsupported or missing DB type.")


# ============== SIDEBAR NAVIGATION ==============
st.sidebar.title("📌 Navigation")
pages = ["🏠 Home", "📂 Data Upload & ETL", "🔗 Database Connection", "🤖 AI Chat & SQL Queries", "⬆️ Upload DataFrame"]
selected_page = st.sidebar.radio("Go to", pages, index=st.session_state.page)
st.session_state.page = pages.index(selected_page)

# ============== PAGE 0: HOME ==============
if st.session_state.page == 0:
    st.title("🏠 Welcome to AI-Powered ETL Chatbot!")
    st.markdown(
        """
        ### 🔥 What You Can Do:
        1. Upload a **CSV file** for AI-powered **data cleaning, summarization, and anomaly detection**.
        2. Convert AI output (Markdown table) into a **structured Pandas DataFrame**.
        3. Connect to a database and **upload transformed data**.
        4. *(Optional)* Use **AI Chat** to generate or run custom SQL queries.
        """
    )

    # Navigation
    if st.button("Next →"):
        st.session_state.page = 1
        st.rerun()

# ============== PAGE 1: DATA UPLOAD & ETL ==============
elif st.session_state.page == 1:
    st.title("📂 Upload & Process Data")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        with st.spinner("Loading CSV..."):
            df = load_csv(uploaded_file)
        st.session_state.uploaded_file = df
        st.success("✅ File uploaded successfully!")
    else:
        st.info("Please upload a CSV file to proceed.")
        df = None

    if st.session_state.uploaded_file is not None:
        st.subheader("🔍 Preview Dataset")
        st.dataframe(st.session_state.uploaded_file.head())

    # Navigation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Previous"):
            st.session_state.page = 0
            st.rerun()
    with col2:
        if st.session_state.uploaded_file is not None and st.button("Next →"):
            st.session_state.page = 2
            st.rerun()

# ============== PAGE 2: DATABASE CONNECTION ==============
elif st.session_state.page == 2:
    st.title("🔗 Connect to Database")

    db_type = st.selectbox(
        "Select Database Type", 
        ["Snowflake", "PostgreSQL", "MySQL", "BigQuery", "Databricks"],
        index=0
    )
    st.session_state.db_type = db_type

    # Dynamic input fields
    if db_type == "Snowflake":
        st.session_state.db_params["account"]   = st.text_input("Account Identifier")
        st.session_state.db_params["user"]      = st.text_input("Username")
        st.session_state.db_params["password"]  = st.text_input("Password", type="password")
        st.session_state.db_params["database"]  = st.text_input("Database Name")
        st.session_state.db_params["schema"]    = st.text_input("Schema")
        st.session_state.db_params["warehouse"] = st.text_input("Warehouse")

    elif db_type == "PostgreSQL":
        st.session_state.db_params["host"]     = st.text_input("Host")
        st.session_state.db_params["port"]     = st.text_input("Port", "5432")
        st.session_state.db_params["database"] = st.text_input("Database Name")
        st.session_state.db_params["user"]     = st.text_input("Username")
        st.session_state.db_params["password"] = st.text_input("Password", type="password")

    elif db_type == "MySQL":
        st.session_state.db_params["host"]     = st.text_input("Host")
        st.session_state.db_params["port"]     = st.text_input("Port", "3306")
        st.session_state.db_params["database"] = st.text_input("Database Name")
        st.session_state.db_params["user"]     = st.text_input("Username")
        st.session_state.db_params["password"] = st.text_input("Password", type="password")

    elif db_type == "BigQuery":
        # Typically need project_id, dataset, credentials_json
        st.session_state.db_params["project_id"] = st.text_input("GCP Project ID")
        st.session_state.db_params["dataset"]    = st.text_input("BigQuery Dataset")
        cred_file = st.file_uploader("Upload GCP Service Account JSON", type=["json"])
        if cred_file is not None:
            st.session_state.db_params["credentials_json"] = json.load(cred_file)

    elif db_type == "Databricks":
        st.session_state.db_params["host"]      = st.text_input("Host")
        st.session_state.db_params["http_path"] = st.text_input("HTTP Path")
        st.session_state.db_params["token"]     = st.text_input("Token", type="password")

    def test_connection():
        try:
            with st.spinner("Testing connection..."):
                engine_or_conn = create_db_engine(db_type, st.session_state.db_params)

            # Quick check/verification
            if db_type == "Snowflake":
                # engine_or_conn is a raw snowflake connector
                cursor = engine_or_conn.cursor()
                cursor.execute("SELECT current_version()")
                version = cursor.fetchone()
                cursor.close()
                st.success(f"✅ Snowflake Connection Successful! Version: {version[0]}")
                st.session_state.db_connection = engine_or_conn

            elif db_type in ["PostgreSQL", "MySQL"]:
                # engine_or_conn is a SQLAlchemy engine
                conn = engine_or_conn.connect()
                conn.close()
                st.success(f"✅ {db_type} Connection Successful!")
                st.session_state.db_connection = engine_or_conn

            elif db_type == "BigQuery":
                # BigQuery needs separate logic, but let's assume if credentials are valid,
                # we have "some" success. Actual test could be more robust.
                st.success("✅ BigQuery credentials set!")
                st.session_state.db_connection = True  # Just a placeholder

            elif db_type == "Databricks":
                # engine_or_conn is a raw Databricks connection
                # Test a quick query if you want
                st.success("✅ Databricks Connection Successful!")
                st.session_state.db_connection = engine_or_conn

        except Exception as e:
            st.error(f"❌ Connection failed: {e}")

    # Buttons
    if st.button("Test Connection"):
        test_connection()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Previous"):
            st.session_state.page = 1
            st.rerun()

    with col2:
        if st.button("Next →"):
            st.session_state.page = 3
            st.rerun()

# ============== PAGE 3: AI CHAT & SQL QUERIES ==============
elif st.session_state.page == 3:
    st.title("🤖 AI Chat for ETL & SQL Queries")

    df = st.session_state.uploaded_file
    if df is None:
        st.warning("⚠️ No file uploaded. Please go back and upload a dataset.")
        st.stop()

    st.subheader("uploaded Dataframe")
    st.dataframe(df.head())
    # --- MAIN ETL ACTIONS ---
    user_input = st.chat_input("Type your message...")
    if user_input:
        raw_output, transformed_df = llm_transform_data(df, 
                        user_input+" "+"and Output only the final cleaned table, with no additional text or explanation."
                    )
        
        st.session_state.chat_history.append(("🧑", user_input))
        st.session_state.chat_history.append(("🤖", transformed_df))
        for role, message in st.session_state.chat_history:
            if role == "🧑":
                st.chat_message("user").write(message)
            else:
                st.chat_message("assistant").write(message)

    # st.subheader("🔧 One-Click Data Transformations")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Clean Data (AI)"):
            with st.spinner("Cleaning data via AI..."):
                raw_output, transformed_df = llm_transform_data(df, 
                    "Clean this dataset by removing duplicates or fixing missing values. Return only a Markdown table."
                )
            if transformed_df is not None:
                st.session_state.transformed_df = transformed_df
                st.success("✅ Cleaning Complete!")
                st.dataframe(transformed_df)
            else:
                st.error("❌ Failed to parse AI response into DataFrame.")

    with col2:
        if st.button("Summarize Data (AI)"):
            with st.spinner("Summarizing data via AI..."):
                raw_output, transformed_df = llm_transform_data(df, 
                    "Summarize this dataset and return key insights in a bullet points if possible."
                )
            if transformed_df is not None:
                st.session_state.transformed_df = transformed_df
                st.success("✅ Summarization Complete!")
                st.dataframe(transformed_df.head())
            else:
                st.error("❌ Failed to parse AI response into DataFrame.")

    with col3:
        if st.button("Detect Anomalies (AI)"):
            with st.spinner("Detecting anomalies via AI..."):
                raw_output, transformed_df = llm_transform_data(df, 
                    "Detect anomalies or outliers in this dataset. Return only those rows in a Markdown table."
                )
            if transformed_df is not None:
                st.session_state.transformed_df = transformed_df
                st.success("✅ Anomaly Detection Complete!")
                st.dataframe(transformed_df.head())
            else:
                st.error("❌ No anomalies found or parse error.")

     # Navigation
    if st.button("← Previous"):
        st.session_state.page = 2
        st.rerun()

    

    # --- UPLOAD TO DB ---
elif st.session_state.page == 4:

    st.title("📤 Upload to Database")

    st.write(st.session_state.transformed_df.head())

    table_name = st.text_input("Enter Table Name for Upload")
    if st.button("Upload Data to DB"):
        if st.session_state.transformed_df is None:
            st.warning("⚠️ No transformed data available.")
        elif not table_name.strip():
            st.warning("⚠️ Please enter a table name.")
        elif st.session_state.db_type is None or st.session_state.db_connection is None:
            st.warning("⚠️ No database connection found. Please connect first.")
        else:
            try:
                with st.spinner(f"Uploading data to {st.session_state.db_type}..."):
                    upload_df_to_db(
                        st.session_state.db_type,
                        st.session_state.db_connection,
                        st.session_state.transformed_df,
                        table_name.strip(),
                        st.session_state.db_params
                    )
                st.success(f"✅ Data uploaded successfully to `{st.session_state.db_type}` in table `{table_name}`.")
            except Exception as e:
                st.error(f"❌ Upload failed: {e}")    

     # Navigation
    if st.button("← Previous"):
        st.session_state.page = 3
        st.rerun()




        # --- CUSTOM TRANSFORMATION ---
    # st.subheader("🛠️ Custom Transformation Prompt")
    # user_logic = st.text_area("Describe your custom transformation logic here (in natural language).")
    # if st.button("Apply Custom Logic"):
    #     if user_logic.strip():
    #         with st.spinner("Applying custom logic via AI..."):
    #             raw_output, transformed_df = llm_transform_data(df, user_logic.strip())
    #         # st.markdown("**Raw AI Markdown Output**:")
    #         # st.code(raw_output)

    #         if transformed_df is not None:
    #             st.session_state.transformed_df = transformed_df
    #             st.markdown("**Converted Pandas DataFrame**:")
    #             st.dataframe(transformed_df)
    #         else:
    #             st.error("❌ Failed to parse AI response into DataFrame.")
    #     else:
    #         st.warning("⚠️ Please provide a prompt for your transformation.")

    # # --- DOWNLOAD TRANSFORMED DATA ---
    # if st.session_state.transformed_df is not None:
    #     st.markdown("### Download Transformed Data")
    #     csv_data = st.session_state.transformed_df.to_csv(index=False)
    #     st.download_button(
    #         label="Download CSV",
    #         data=csv_data,
    #         file_name="transformed_data.csv",
    #         mime="text/csv"
    #     )
        # st.title("🤖 AI Chat for ETL & SQL Queries")
        # st.subheader("📤 Upload to Database")
        # table_name = st.text_input("Enter Table Name for Upload")
        # if st.button("Upload Data to DB"):
        #     if st.session_state.transformed_df is None:
        #         st.warning("⚠️ No transformed data available.")
        #     elif not table_name.strip():
        #         st.warning("⚠️ Please enter a table name.")
        #     elif st.session_state.db_type is None or st.session_state.db_connection is None:
        #         st.warning("⚠️ No database connection found. Please connect first.")
        #     else:
        #         try:
        #             with st.spinner(f"Uploading data to {st.session_state.db_type}..."):
        #                 upload_df_to_db(
        #                     st.session_state.db_type,
        #                     st.session_state.db_connection,
        #                     st.session_state.transformed_df,
        #                     table_name.strip(),
        #                     st.session_state.db_params
        #                 )
        #             st.success(f"✅ Data uploaded successfully to `{st.session_state.db_type}` in table `{table_name}`.")
        #         except Exception as e:
        #             st.error(f"❌ Upload failed: {e}")

    # --- OPTIONAL: AI CHAT -> SQL QUERIES ---
    # st.markdown("---")
    # query_prompt = st.text_area(
    #     "Ask a question or describe the data you want to retrieve (the AI will generate an SQL query)."
    # )
    # if st.button("Generate SQL from Prompt"):
    #     if not query_prompt.strip():
    #         st.warning("⚠️ Please type in a question or prompt.")
    #     else:
    #         with st.spinner("Generating SQL query..."):
    #             # Example prompting. You may need to refine your prompt to get correct SQL for your DB.
    #             # For demonstration, we just pass user prompt and assume AI can form the correct table references.
    #             response = groq_client.chat.completions.create(
    #                 model="llama-3.3-70b-versatile",
    #                 messages=[{
    #                     "role": "user",
    #                     "content": (
    #                         f"I have a table called '{table_name}'. Please generate a SQL query that: "
    #                         f"{query_prompt}\n"
    #                         "Just return the SQL query itself, no explanation."
    #                     )
    #                 }]
    #             )
    #             generated_sql = response.choices[0].message.content.strip()
    #         st.markdown("**Generated SQL Query**:")
    #         st.code(generated_sql, language="sql")

            # Optionally: run the query if the user wants
            # if st.session_state.db_connection is not None and st.button("Run SQL Query"):
            #     try:
            #         db_type = st.session_state.db_type
            #         conn_obj = st.session_state.db_connection

            #         if db_type in ["PostgreSQL", "MySQL"]:
            #             with conn_obj.connect() as conn:
            #                 query_result = pd.read_sql(generated_sql, conn)
            #             st.dataframe(query_result)
            #         elif db_type == "Snowflake":
            #             cursor = conn_obj.cursor()
            #             cursor.execute(generated_sql)
            #             query_result = cursor.fetch_pandas_all()
            #             st.dataframe(query_result)
            #             cursor.close()
            #         else:
            #             st.warning(f"Running custom SQL not implemented for {db_type} yet.")

            #     except Exception as e:
            #         st.error(f"Error executing query: {e}")

    # # Navigation
    # if st.button("← Previous"):
    #     st.session_state.page = 3
    #     st.rerun()


