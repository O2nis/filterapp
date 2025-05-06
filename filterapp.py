import streamlit as st
import pandas as pd
from thefuzz import process, fuzz
from io import StringIO, BytesIO
import chardet
import uuid
import openpyxl

# Algorithm options with descriptions for matching function
MATCH_ALGORITHMS = {
    "Ratio": fuzz.ratio,
    "Partial Ratio": fuzz.partial_ratio,
    "Token Sort": fuzz.token_sort_ratio,
    "Token Set": fuzz.token_set_ratio,
    "WRatio": fuzz.WRatio
}

def detect_encoding(file):
    """Detect file encoding using chardet for CSV files"""
    rawdata = file.read()
    result = chardet.detect(rawdata)
    file.seek(0)  # Reset file pointer
    return result['encoding']

def read_file(file):
    """Read CSV or Excel file with automatic encoding detection for CSV"""
    file_ext = file.name.split('.')[-1].lower()
    
    if file_ext in ['xlsx', 'xls']:
        try:
            return pd.read_excel(file, engine='openpyxl' if file_ext == 'xlsx' else 'xlrd')
        except Exception as e:
            st.error(f"Error reading Excel file: {str(e)}")
            return None
    else:  # Assume CSV
        encoding = detect_encoding(file)
        try:
            return pd.read_csv(file, encoding=encoding)
        except UnicodeDecodeError:
            file.seek(0)
            try:
                return pd.read_csv(file, encoding='utf-16')
            except:
                file.seek(0)
                return pd.read_csv(file, encoding='latin1')  # Fallback

def fill_missing_values(df1, df2, match_col1, match_col2, data_col2, threshold=80, algorithm="Token Set", prevent_duplicates=True):
    """Fill missing values using fuzzy matching on selected columns"""
    filled_df = df1[[match_col1]].copy()  # Start with only the matching column
    scorer = MATCH_ALGORITHMS[algorithm]
    
    # Add result columns
    filled_df['Filled Data'] = None
    filled_df['Match Score'] = 0
    filled_df['Matched Name'] = ""
    filled_df['Algorithm Used'] = algorithm
    
    # Create dictionary from df2 using selected matching and data columns
    df2_dict = dict(zip(
        df2[match_col2],
        df2[data_col2]
    ))
    
    used_matches = set() if prevent_duplicates else None
    
    for index, row in filled_df.iterrows():
        if pd.isna(row.get('Filled Data', True)):  # Check if data needs filling
            available_matches = {k:v for k,v in df2_dict.items() 
                               if not prevent_duplicates or k not in used_matches}
            
            if not available_matches:
                continue
                
            best_match = process.extractOne(
                row[match_col1],
                available_matches.keys(),
                scorer=scorer
            )
            
            if best_match and best_match[1] >= threshold:
                match_name, match_score = best_match[0], best_match[1]
                filled_df.at[index, 'Filled Data'] = df2_dict[match_name]
                filled_df.at[index, 'Match Score'] = match_score
                filled_df.at[index, 'Matched Name'] = match_name
                
                if prevent_duplicates:
                    used_matches.add(match_name)
    
    return filled_df

def filter_data(filter_df, data_df, filter_column, keep_matching=True):
    """Filter data based on values in a column, either keeping or removing matches"""
    filter_values = filter_df[filter_column].unique()
    if keep_matching:
        filtered_df = data_df[data_df[filter_column].isin(filter_values)]
    else:
        filtered_df = data_df[~data_df[filter_column].isin(filter_values)]
    return filtered_df

def split_column_values(df, column, delimiter):
    """Split values in a column based on delimiter into separate columns"""
    # Convert column to string and split
    split_data = df[column].astype(str).str.split(delimiter, expand=True)
    
    # Create new column names
    max_splits = split_data.shape[1]
    new_columns = [f"{column}_part_{i+1}" for i in range(max_splits)]
    
    # Create new dataframe with split columns
    result_df = pd.DataFrame(split_data.values, columns=new_columns)
    
    # Include original columns (except the split column) and append split columns
    other_columns = [col for col in df.columns if col != column]
    final_df = pd.concat([df[other_columns], result_df], axis=1)
    
    return final_df

def to_excel(df):
    """Convert DataFrame to Excel bytes"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

def main():
    st.title("📊 Data Processing Tool")
    st.markdown("Select between three functions: fuzzy matching, filtering, or column splitting")
    
    # Function selector
    function_choice = st.radio(
        "Select function:",
        ["🔍 Fuzzy Match Values", "🔎 Filter Data", "✂️ Split Column Values"],
        horizontal=True
    )
    
    if function_choice == "🔍 Fuzzy Match Values":
        st.header("Fuzzy Match Values")
        st.markdown("Fill missing values by matching selected columns between CSV or Excel files")
        
        col1, col2 = st.columns(2)
        with col1:
            uploaded_file1 = st.file_uploader(
                "File with missing values", 
                type=["csv", "xlsx", "xls"],
                help="Supports CSV (UTF-8, UTF-16, etc.) and Excel (.xlsx, .xls)",
                key="match_file1"
            )
        with col2:
            uploaded_file2 = st.file_uploader(
                "Reference file with complete values", 
                type=["csv", "xlsx", "xls"],
                key="match_file2"
            )
        
        if uploaded_file1 and uploaded_file2:
            # Read files
            df1 = read_file(uploaded_file1)
            df2 = read_file(uploaded_file2)
            
            if df1 is None or df2 is None:
                return
            
            with st.expander("Column Selection and Matching Settings", expanded=True):
                # Column selection for first file
                match_col1 = st.selectbox(
                    "Select column to match from first file",
                    options=df1.columns,
                    key="match_col1"
                )
                
                # Column selection for second file
                match_col2 = st.selectbox(
                    "Select column to match from second file",
                    options=df2.columns,
                    key="match_col2"
                )
                
                data_col2 = st.selectbox(
                    "Select column with data to fill from second file",
                    options=df2.columns,
                    key="data_col2"
                )
                
                # Matching settings
                algorithm = st.selectbox(
                    "Matching Algorithm",
                    options=list(MATCH_ALGORITHMS.keys()),
                    index=3
                )
                
                threshold = st.slider(
                    "Minimum Match Score",
                    min_value=0, max_value=100, value=80
                )
                
                prevent_duplicates = st.checkbox(
                    "Prevent duplicate use of reference values",
                    value=True
                )
                
                output_format = st.selectbox(
                    "Output file format",
                    options=['CSV', 'Excel'],
                    index=0
                )
                
                if output_format == 'CSV':
                    output_encoding = st.selectbox(
                        "Output file encoding",
                        options=['utf-8', 'utf-16'],
                        index=0
                    )
                else:
                    output_encoding = None
            
            try:
                # Clean data for selected columns
                df1[match_col1] = df1[match_col1].astype(str).str.lower().str.strip()
                df2[match_col2] = df2[match_col2].astype(str).str.lower().str.strip()
                
                # Process data
                with st.spinner("Matching values..."):
                    result_df = fill_missing_values(
                        df1, df2,
                        match_col1=match_col1,
                        match_col2=match_col2,
                        data_col2=data_col2,
                        threshold=threshold,
                        algorithm=algorithm,
                        prevent_duplicates=prevent_duplicates
                    )
                
                st.success(f"Completed! Filled {result_df['Match Score'].gt(0).sum()} values")
                
                tab1, tab2 = st.tabs(["All Data", "Filled Values"])
                with tab1:
                    st.dataframe(result_df)
                with tab2:
                    st.dataframe(result_df[result_df['Match Score'] > 0])
                
                # Download results
                if output_format == 'Excel':
                    excel_data = to_excel(result_df)
                    st.download_button(
                        "Download Results",
                        data=excel_data,
                        file_name="filled_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:  # CSV
                    csv = result_df.to_csv(index=False)
                    if output_encoding == 'utf-16':
                        csv = csv.encode('utf-16')
                    else:
                        csv = csv.encode('utf-8')
                    st.download_button(
                        "Download Results",
                        data=csv,
                        file_name=f"filled_results_{output_encoding}.csv",
                        mime="text/csv"
                    )
                
            except Exception as e:
                st.error(f"Error processing files: {str(e)}")
    
    elif function_choice == "🔎 Filter Data":
        st.header("Filter Data")
        st.write("Filter rows from a second file based on values from a column in a first file.")
        
        # Upload the first file (filter criteria)
        st.subheader("Step 1: Upload the file with filter values")
        filter_file = st.file_uploader(
            "Upload your filter file", 
            type=['csv', 'xlsx', 'xls'], 
            key="filter_file"
        )
        
        if filter_file is not None:
            try:
                filter_df = read_file(filter_file)
                if filter_df is None:
                    return
                st.success("Filter file loaded successfully!")
                
                # Select the column to use for filtering
                filter_column = st.selectbox(
                    "Select the column from the filter file to use for matching",
                    filter_df.columns,
                    key="filter_column"
                )
                
                # Filter mode selection
                filter_mode = st.radio(
                    "Filter mode:",
                    ["Keep matching rows", "Remove matching rows"],
                    index=0,
                    key="filter_mode"
                )
                keep_matching = filter_mode == "Keep matching rows"
                
                # Get unique values from the selected column
                filter_values = filter_df[filter_column].unique()
                st.write(f"Found {len(filter_values)} unique values in the selected column.")
                
                # Upload the second file (data to filter)
                st.subheader("Step 2: Upload the file to filter")
                data_file = st.file_uploader(
                    "Upload the file to filter", 
                    type=['csv', 'xlsx', 'xls'], 
                    key="data_file"
                )
                
                if data_file is not None:
                    try:
                        data_df = read_file(data_file)
                        if data_df is None:
                            return
                        st.success("Data file loaded successfully!")
                        
                        # Check if the filter column exists in the data file
                        matching_columns = [col for col in data_df.columns if col == filter_column]
                        
                        if not matching_columns:
                            st.warning(f"No column named '{filter_column}' found in the data file.")
                            st.write("Available columns in data file:", data_df.columns.tolist())
                        else:
                            # Filter the data
                            with st.spinner("Filtering data..."):
                                filtered_df = filter_data(filter_df, data_df, filter_column, keep_matching)
                            
                            st.subheader("Filtering Results")
                            st.write(f"Original data rows: {len(data_df)}")
                            st.write(f"Filtered data rows: {len(filtered_df)}")
                            
                            if len(filtered_df) > 0:
                                # Show filtered data
                                st.dataframe(filtered_df)
                                
                                # Output format selection
                                output_format = st.selectbox(
                                    "Output file format",
                                    options=['CSV', 'Excel'],
                                    index=0,
                                    key="filter_output_format"
                                )
                                
                                # Download button
                                if output_format == 'Excel':
                                    excel_data = to_excel(filtered_df)
                                    st.download_button(
                                        "Download Filtered Data",
                                        excel_data,
                                        "filtered_data.xlsx",
                                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        key='download-excel'
                                    )
                                else:
                                    csv = filtered_df.to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        "Download Filtered Data",
                                        csv,
                                        "filtered_data.csv",
                                        "text/csv",
                                        key='download-csv'
                                    )
                            else:
                                st.warning("No matching rows found.")
                                
                    except Exception as e:
                        st.error(f"Error loading data file: {str(e)}")
                        
            except Exception as e:
                st.error(f"Error loading filter file: {str(e)}")
    
    else:  # Split Column Values
        st.header("Split Column Values")
        st.markdown("Split text in a selected column into multiple columns based on a delimiter")
        
        # Upload file
        st.subheader("Step 1: Upload the file to process")
        split_file = st.file_uploader(
            "Upload your file",
            type=['csv', 'xlsx', 'xls'],
            key="split_file"
        )
        
        if split_file is not None:
            try:
                df = read_file(split_file)
                if df is None:
                    return
                st.success("File loaded successfully!")
                
                # Select column to split
                split_column = st.selectbox(
                    "Select the column to split",
                    df.columns,
                    key="split_column"
                )
                
                # Input delimiter
                delimiter = st.text_input(
                    "Enter the delimiter (e.g., -, ,, ;)",
                    value="-",
                    key="delimiter"
                )
                
                if delimiter:
                    try:
                        # Process the split
                        with st.spinner("Splitting column values..."):
                            result_df = split_column_values(df, split_column, delimiter)
                        
                        st.subheader("Split Results")
                        st.write(f"Original columns: {len(df.columns)}")
                        st.write(f"New columns: {len(result_df.columns)}")
                        
                        # Show results
                        st.dataframe(result_df)
                        
                        # Download results as CSV
                        csv = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download Split Results",
                            csv,
                            "split_results.csv",
                            "text/csv",
                            key='download-split-csv'
                        )
                        
                    except Exception as e:
                        st.error(f"Error processing split: {str(e)}")
                        
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

if __name__ == "__main__":
    main()
