#24471918
#CITS1401 Python Project 2
#Clayton de Wit

def main(CSVfile, TXTfile, category):
    # Open the CSVfile and TXTfile
    try:
        with open(CSVfile, "r") as open_CSV, open(TXTfile, "r") as open_TXT:
            content_CSV = open_CSV.read()
            content_TXT = open_TXT.read()
    except (FileNotFoundError, IOError, PermissionError):
        return None, None, None, None

    # Check if the file contents are valid
    if not content_CSV or not content_TXT:
        return None, None, None, None

    # Process CSV data
    CSV_lines = content_CSV.splitlines()
    all_data = [line.split(',') for line in CSV_lines]

    # Process headers and locate indices
    header_row = [element.lower().strip() for element in all_data[0]]
    try:
        country_index = header_row.index('country')
        hospital_id_index = header_row.index('hospital_id')
        deaths_2022_index = header_row.index('no_of_deaths_in_2022')
        category_index = header_row.index('hospital_category')
        staff_count_index = header_row.index('no_of_staff')
        female_patients_index = header_row.index('female_patients')
        deaths_2023_index = header_row.index('no_of_deaths_in_2023')
    except ValueError:
        print("Missing required headers. Please check your CSV file.")
        return None, None, None, None  # Ensure we return 4 values

    all_data = all_data[1:]  # Remove header row
    if not all_data:
        print("No data found in CSV file.")
        return None, None, None, None  # Ensure we return 4 values

    #check for valid data and collect valid entries
    valid_data = []
    for row in all_data:
        try:
            # Convert relevant columns to the correct types
            no_of_deaths_2022 = int(row[deaths_2022_index].strip())
            no_of_staff = int(row[staff_count_index].strip())
            female_patients = int(row[female_patients_index].strip())
            no_of_deaths_2023 = int(row[deaths_2023_index].strip())

            # Check for invalid data
            if (no_of_deaths_2022 < 0 or no_of_staff <= 0 or female_patients < 0 or no_of_deaths_2023 < 0):
                continue
            valid_data.append(row) # Add valid row to the list

        except (ValueError, IndexError):
            print("Error processing row. Skipping this row.")
            continue

    # Check if there are any valid data entries
    if not valid_data:
        print("No valid data found in the CSV file.")
        return None, None, None, None  # Ensure we return 4 values

    # Process TXT data
    lines = content_TXT.strip().split('\n')
    TXT_data = [line.strip().split(', ') for line in lines]

    Output_1 = country_hospital_data(valid_data, country_index, hospital_id_index, deaths_2022_index, TXT_data)
    Output_2 = cosine_similarity(valid_data, country_index, deaths_2022_index, TXT_data)
    Output_3 = cancer_admit_var(valid_data, TXT_data, category, country_index, hospital_id_index, category_index)
    Output_4 = hospital_category_statistics(valid_data, category, country_index, category_index, female_patients_index, staff_count_index, deaths_2022_index, deaths_2023_index)

    return Output_1, Output_2, Output_3, Output_4


#defining Output_1 "country_hospital_data"
def country_hospital_data(valid_data, country_index, hospital_id_index, deaths_2022_index, TXT_data):
    country_ids = {}
    country_deaths_2022 = {}
    covid_stroke_admissions = {}

    # Process valid_data
    for row in valid_data:
        if len(row) <= max(country_index, hospital_id_index, deaths_2022_index):
            continue  # Skip rows with insufficient columns
        country = row[country_index].lower().strip()
        hospital_id = row[hospital_id_index].lower().strip()

        # Ensure the deaths data is a valid integer
        try:
            deaths = int(row[deaths_2022_index].strip())
        except ValueError:
            continue  # Skip invalid deaths data

        if country not in country_ids:
            country_ids[country] = []
            country_deaths_2022[country] = []

        country_ids[country].append(hospital_id)
        country_deaths_2022[country].append(deaths)

    # Process TXT data
    for entry in TXT_data:
        country = entry[0].split(':')[1].strip().lower()  # Extract country from TXT_data

        # Extract COVID and Stroke admission values
        try:
            covid_admit = int(entry[2].split(':')[1].strip())
            stroke_admit = int(entry[3].split(':')[1].strip())
        except (IndexError, ValueError):
            continue  # Skip invalid entries

        total = covid_admit + stroke_admit
        
        if country in covid_stroke_admissions:
            covid_stroke_admissions[country].append(total)
        else:
            covid_stroke_admissions[country] = [total]

    return country_ids, country_deaths_2022, covid_stroke_admissions
    
#defining Output_2 the "cosine_similarity"
def cosine_similarity(valid_data, country_index, deaths_2022_index, TXT_data):
    combined_data = {}  # Creating a blank dictionary

    # Process valid_data to extract deaths
    for elements in valid_data:
        country = elements[country_index].lower()  # Get country name
        try:
            deaths = int(elements[deaths_2022_index].strip())  # Get deaths count
        except (ValueError, IndexError):
            continue  # Skip invalid deaths data

        if country not in combined_data:
            combined_data[country] = {'deaths': [], 'admissions': []}
        combined_data[country]['deaths'].append(deaths)  # Append to dictionary

    # Process TXT_data to extract COVID and stroke admissions
    for entry in TXT_data:
        country = entry[0].split(':')[1].strip().lower()  # Extract country
        try:
            covid_admit = int(entry[2].split(':')[1].strip())  # Extract COVID admissions
            stroke_admit = int(entry[3].split(':')[1].strip())  # Extract Stroke admissions
        except (IndexError, ValueError):
            continue  # Skip invalid entries

        total = covid_admit + stroke_admit  # Calculate total admissions

        if country in combined_data:
            combined_data[country]['admissions'].append(total)  # Append to existing entries
        else:
            combined_data[country] = {'deaths': [], 'admissions': [total]}  # Create new entry

    # Calculate the normalized cosine similarity for each country
    cosine_s_dictionary = {}

    for country, values in combined_data.items():
        deaths = values['deaths']
        admissions = values['admissions']

        if deaths and admissions:  # Ensure both lists have data
            # Calculate the dot product
            dot_product = sum(d * a for d, a in zip(deaths, admissions))

            # Calculate sums of squares
            sum_deaths_squared = sum(d ** 2 for d in deaths)
            sum_admissions_squared = sum(a ** 2 for a in admissions)

            # Calculate the normalization factor (square root of the sums of squares)
            normalization_factor = (sum_deaths_squared ** 0.5) * (sum_admissions_squared ** 0.5)

            if normalization_factor != 0:  # Prevent division by zero
                normalized_result = dot_product / normalization_factor
                cosine_s_dictionary[country] = round(normalized_result, 4)  # Round to 4 decimal places
            else:
                cosine_s_dictionary[country] = 0  # Handle case where normalization factor is zero

    return cosine_s_dictionary

#defining Output_3 the "variance of patients admitted to a hospital due to cancer in a category/country"
def cancer_admit_var(valid_data, TXT_data, category, country_index, hospital_id_index, category_index):
    admissions_dict = {}

    # Create a dictionary to map hospital IDs to cancer admissions from TXT_data
    cancer_admissions_dict = {}
    for entry in TXT_data:
        try:
            country = entry[0].split(':')[1].strip().lower()  # Extract country name
            hospital_id = entry[1].split(':')[1].strip()  # Extract hospital ID
            cancer_admissions = int(entry[4].split(':')[1].strip())  # Extract cancer admissions

            if hospital_id not in cancer_admissions_dict:  # Check if ID is in dictionary
                cancer_admissions_dict[hospital_id] = (country, [])

            cancer_admissions_dict[hospital_id][1].append(cancer_admissions)  # Append to dictionary
        except (IndexError, ValueError):
            continue  # Skip invalid entries

    # Process valid_data to extract admissions for the specified category
    for elements in valid_data:  # Use only valid data
        country = elements[country_index].lower()
        hospital_id = elements[hospital_id_index].strip()  # Hospital ID in valid_data
        category_value = elements[category_index].lower()  # Get category value

        if category_value == category.lower() and hospital_id in cancer_admissions_dict:
            # Get cancer admissions for the current hospital ID
            matched_country, admissions = cancer_admissions_dict[hospital_id]
            if country == matched_country:
                if country not in admissions_dict:
                    admissions_dict[country] = []  # Initialize if not already present
                admissions_dict[country].extend(admissions)  # Append all admissions

    # Calculate variance for each country
    variance_dict = {}
    for country, admissions in admissions_dict.items():
        n = len(admissions)
        if n > 1:  # Variance requires at least 2 data points
            mean = sum(admissions) / n
            variance = sum((x - mean) ** 2 for x in admissions) / (n - 1)  # Sample variance
            variance_dict[country] = round(variance, 4)  # Round to 4 decimal places
        else:
            variance_dict[country] = 0.0  # Handle cases with insufficient data

    return variance_dict

#defining Output_4 "hospital_category_statistics"
def hospital_category_statistics(valid_data, category, country_index, category_index, female_patients_index, staff_count_index, deaths_2022_index, deaths_2023_index):
    category_country_dict = {}
    category_cont_dict = {}

    # Process valid_data to gather statistics for the specified category
    for elements in valid_data:
        try:
            country = elements[country_index].lower()
            category_value = elements[category_index].lower()
            female_patients = int(elements[female_patients_index])  # Variable locations in valid_data
            staff_count = int(elements[staff_count_index])
            deaths_2022 = int(elements[deaths_2022_index])
            deaths_2023 = int(elements[deaths_2023_index])

            # Initialize the category in the outer dictionary if not present
            if category_value not in category_country_dict:
                category_country_dict[category_value] = {}
                category_cont_dict[category_value] = {}

            # Initialize the country entry in the inner dictionary if not present
            if country not in category_country_dict[category_value]:
                category_cont_dict[category_value][country] = [0, 0, 0]
                category_country_dict[category_value][country] = {
                    'female_patients_sum': 0,
                    'staff_counts': [],  # Store staff counts in a list to find max
                    'total_deaths_2022': 0,
                    'total_deaths_2023': 0,
                    'count': 0}

            # Update the statistics
            country_stats = category_country_dict[category_value][country]
            country_stats['female_patients_sum'] += female_patients
            country_stats['staff_counts'].append(staff_count)  # Append staff count
            country_stats['total_deaths_2022'] += deaths_2022
            country_stats['total_deaths_2023'] += deaths_2023
            country_stats['count'] += 1

        except (IndexError, ValueError):
            continue  # Skip rows with invalid data

    # Calculate averages and store results
    for category_value, countries in category_country_dict.items():
        for country, stats in countries.items():
            average_female_patients = (
                stats['female_patients_sum'] / stats['count'] if stats['count'] > 0 else 0) 
            average_deaths_2022 = (
                stats['total_deaths_2022'] / stats['count'] if stats['count'] > 0 else 0)
            average_deaths_2023 = (
                stats['total_deaths_2023'] / stats['count'] if stats['count'] > 0 else 0)
            max_staff = max(stats['staff_counts']) if stats['staff_counts'] else 0 # Calculate maximum staff
            
            # Calculate percentage change in deaths
            percentage_change = (
                ((average_deaths_2023 - average_deaths_2022) / average_deaths_2022) * 100
                if average_deaths_2022 != 0 else 0)

            # Store results in the inner dictionary
            category_cont_dict[category_value][country][0] = round(average_female_patients, 4)   # Average female patients
            category_cont_dict[category_value][country][1] = max_staff                           # Maximum staff
            category_cont_dict[category_value][country][2] = round(percentage_change, 4)         # Percentage change

    return category_cont_dict