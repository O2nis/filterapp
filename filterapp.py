import streamlit as st
import pandas as pd

def main():
    st.title("CSV Filtering Tool")
    st.write("This app filters rows from a second CSV file based on values from a column in a first CSV file.")
    
    # Upload the first CSV (filter criteria)
    st.header("Step 1: Upload the CSV with filter values")
    filter_file = st.file_uploader("Upload your filter CSV file", type=['csv'], key="filter_file")
    
    if filter_file is not None:
        try:
            filter_df = pd.read_csv(filter_file)
            st.success("Filter CSV loaded successfully!")
            
            # Select the column to use for filtering
            filter_column = st.selectbox(
                "Select the column from the filter CSV to use for matching",
                filter_df.columns,
                key="filter_column"
            )
            
            # Get unique values from the selected column
            filter_values = filter_df[filter_column].unique()
            st.write(f"Found {len(filter_values)} unique values in the selected column.")
            
            # Upload the second CSV (data to filter)
            st.header("Step 2: Upload the CSV to filter")
            data_file = st.file_uploader("Upload the CSV file to filter", type=['csv'], key="data_file")
            
            if data_file is not None:
                try:
                    data_df = pd.read_csv(data_file)
                    st.success("Data CSV loaded successfully!")
                    
                    # Check if the filter column exists in the data CSV
                    matching_columns = [col for col in data_df.columns if col == filter_column]
                    
                    if not matching_columns:
                        st.warning(f"No column named '{filter_column}' found in the data CSV.")
                        st.write("Available columns in data CSV:", data_df.columns.tolist())
                    else:
                        # Filter the data
                        filtered_df = data_df[data_df[filter_column].isin(filter_values)]
                        
                        st.header("Filtering Results")
                        st.write(f"Original data rows: {len(data_df)}")
                        st.write(f"Filtered data rows: {len(filtered_df)}")
                        
                        if len(filtered_df) > 0:
                            # Show filtered data
                            st.dataframe(filtered_df)
                            
                            # Download button
                            csv = filtered_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "Download Filtered CSV",
                                csv,
                                "filtered_data.csv",
                                "text/csv",
                                key='download-csv'
                            )
                        else:
                            st.warning("No matching rows found.")
                            
                except Exception as e:
                    st.error(f"Error loading data CSV: {str(e)}")
                    
        except Exception as e:
            st.error(f"Error loading filter CSV: {str(e)}")

if __name__ == "__main__":
    main()