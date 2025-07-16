from flight_querying import query_flights
from weather_querying import query_weather
import plotly.graph_objects as go
from dotenv import load_dotenv
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
import pickle 
from Aniket_Stages.feature_eng import add_features, add_altitude, add_smoothed_RoC, add_RoC, add_smoothed_alt, add_rolling_mean

# Function -------------------------------------------------------------------------------------------------------------------------------------------------------
def create_mapbox_map_per_flight(flight_id: int):
    """ 
    Function takes a flight id and uses the Mapbox mapping feature from plotly and a MAPBOX_PUBLIC_TOKEN to create a map of the fligt's 
    latitude and longitude coordinates.

    Parameter:
        flight_id: A integer number corresponding with the flight map you want.

    Returns:
        fig: A plotly figure object with Scattermapbox functionality
        latitude: A numpy array of Double data type latitudes for the flight.
        longitude: A numpy array of Double data type longitudes for the flight.
    """

    # Load .env file
    load_dotenv()

    # Get mapbox token to run the code.
    mapbox_access_token = os.getenv('MAPBOX_PUBLIC_TOKEN')

    # Specify Waterloo Wellington Flight Center coordinates and specify the columns to query
    wwfc_lat = 43.45567935107457
    wwfc_lon = -80.3881582036048
    query_columns = ['lat', 'lng']

    # In the event that nothing is sent. Then set a basic
    if flight_id == "":
        # Fine-tune lat and long on the DataFrame
        latitude = [43.45567935107457]
        longitude = [-80.3881582036048]
    else:
        # Get Flight data for specific flight.
        query_conn = query_flights()
        query_result = query_conn.get_flight_data_on_id(query_columns, flight_id)
        query_result.replace(0, np.nan, inplace=True)

        # Filter rows where lat and lng are within Canada bounds
        canada_bounds = {'lat_min': 41.676555, 'lat_max': 83.110626, 'lng_min': -141.00187, 'lng_max': 82.617592}

        query_result = query_result[
            (query_result['lat'] >= canada_bounds['lat_min']) & (query_result['lat'] <= canada_bounds['lat_max']) &
            (query_result['lng'] >= canada_bounds['lng_min']) & (query_result['lng'] <= canada_bounds['lng_max'])
        ]

        # Fine-tune lat and long on the DataFrame
        latitude = query_result["lat"].to_numpy()
        longitude = query_result["lng"].to_numpy() * (-1)

    # Graphing
    fig = go.Figure(go.Scattermapbox(
        lat=latitude,
        lon=longitude,
        mode="lines",
        marker=go.scattermapbox.Marker(
            size=5
        ),
        text=['WWFC'],
    ))

    # Update the figure layout
    fig.update_layout(
        hovermode='closest',
        margin=dict(l=5, r=5, t=5, b=5),
        mapbox=dict(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=wwfc_lat,
                lon=wwfc_lon
            ),
            pitch=0,
            zoom=12,
        )
    )

    return fig, latitude, longitude


# Function -------------------------------------------------------------------------------------------------------------------------------------------------------
def soc_graph(flight_ids: list):
    """
    The function takes in flight ids and dates and creates a single matplotlib figure line graph of soc vs. time with warning and danger zones. 

    Parameters:
        flight_ids: A list of all flight ids form the DB. Index should corresponds with the flight_dates index.
        flight_dates: A list of all flight dates form the DB. Index should corresponds with the flight_ids index.

    Returns:
        soc_ax: The matplotlib figure axis with stored soc vs. time graph data and other supports.
    """

    # Make flight db connection
    flight_db_conn = query_flights()
    flight_data = flight_db_conn.get_flight_soc_and_time(flight_ids)

    # Set warning zone, and danger zone ranges
    x_zone = [0, 0, 60, 60]
    warning_zone = [15.01, 30, 30, 15.01]
    danger_zone = [0, 15, 15, 0]

    # Set Plot
    soc_figure = plt.figure(figsize=(5, 8), dpi = 110)

    # Fill ranges
    plt.fill(x_zone, warning_zone, c="gold", alpha=0.5)
    plt.fill(x_zone, danger_zone, c='r', alpha=0.6)

    # Add text to fill ranges
    plt.text(0.2, 20.5, '  Warning', fontweight='bold')
    plt.text(0, 5.5, '  Danger', fontweight='bold', c='white')

    # Plot the graphs
    for i in range(0, len(flight_ids)):

        # Define the date and id
        id = flight_ids[i]

        # Get the soc and time and plot it with a legend label
        soc = flight_data[id]['soc']
        time = flight_data[id]['time_min']
        date = flight_data[id]["date"]
        plt.plot(time, soc, label=date)
    
    # Add labels and legend to plot
    plt.xlim([0, 55])
    plt.ylim([0, 101])
    plt.xlabel("time (min)")
    plt.ylabel("SOC")
    plt.title("Time vs SOC")
    
    # plt.legend(loc="lower left")
    plt.legend(loc='lower left', fontsize="9", bbox_to_anchor= (0, -0.2), ncol=4,
            borderaxespad=0, frameon=False)

    return soc_figure


# Function -------------------------------------------------------------------------------------------------------------------------------------------------------
def power_graph(flight_ids: list):
    """
    The function takes in flight ids and dates and creates a single matplotlib figure scatter graph of motor_power vs. time. 

    Parameters:
        flight_ids: A list of all flight ids form the DB. Index should corresponds with the flight_dates index.
        flight_dates: A list of all flight dates form the DB. Index should corresponds with the flight_ids index.

    Returns:
        power_ax: The matplotlib figure axis with stored motor_power vs. time graph data and other supports.
    """

    # Make flight db connection
    flight_db_conn = query_flights()
    flight_data = flight_db_conn.get_flight_motor_power_and_time(flight_ids)

    # Set Plot
    power_figure = plt.figure(figsize=(6, 8), dpi= 110)

    # Plot the graphs
    for i in range(0, len(flight_ids)):

        # Define the date and id
        id = flight_ids[i]
 
        # Get the soc and time and plot it with a legend label
        motor_power = flight_data[id]['motor_power']
        time = flight_data[id]['time_min']
        date = flight_data[id]["date"]
        plt.plot(time, motor_power, label=date)
    
    # Add labels and legend to plot
    plt.xlim([0, 55])
    plt.ylim([0, 70])
    plt.xlabel("time (min)")
    plt.ylabel("Motor power (kilowatts-KW)")
    plt.title("Time vs Motor power")
    
    # plt.legend(loc="lower left")
    plt.legend(loc='lower left', fontsize="9", bbox_to_anchor= (0, -0.2), ncol=4,
            borderaxespad=0, frameon=False)

    return power_figure


# Function -------------------------------------------------------------------------------------------------------------------------------------------------------
def power_soc_rate_scatterplot(flight_id: list, activities_filter: list):
    """
    The function takes in flight ids and dates and creates a single matplotlib figure scatter plot of motor_power vs. SOC rate of change.
    A legend of activities is also included. 
    Parameters:
        flight_ids: A list of all flight ids form the DB. Index should corresponds with the flight_dates index.
        flight_dates: A list of all flight dates form the DB. Index should corresponds with the flight_ids index.
        activities_filter: A list of all flight activities the user would like to filter by.
    Returns:
        scatter_ax: The matplotlib figure axis with stored scatter plot data and other supports.
    """

    # Set Plot
    scatter_figure = plt.figure(figsize=(6, 6))
    scatter_ax = scatter_figure.add_subplot(1, 1, 1)
    scatter_figure.tight_layout()

    # Only plot if id is there.
    if flight_id != "":

        # Make flight db connection
        flight_db_conn = query_flights()
        flight_data = flight_db_conn.get_flight_power_soc_rate(flight_id, activities_filter)

        # Get the motor power and soc rate
        motor_power = flight_data[flight_id]['motor_power']
        soc_rate_of_change = flight_data[flight_id]['soc_rate_of_change']

        # Get activities
        activity = flight_data[flight_id]['activity']

        # Determine unique activities and assign colors or markers
        unique_activities = np.unique(activity)

        # gets colours from the jet colour map
        colors = plt.cm.jet(np.linspace(0, 1, len(unique_activities))) 

        # assigns each unique activity to a colour
        activity_color_map = dict(zip(unique_activities, colors)) 
        
        # Iterate over each unique activity type
        for act in unique_activities:
            
            # Create a boolean mask where the condition (activity == act) is True
            # This mask is used to select only the data points corresponding to the current activity
            act_mask = activity == act

            # Plot the scatter points for the current activity
            # motor_power[act_mask] and soc_rate_of_change[act_mask] select the data points that correspond to the current activity
            plt.scatter(motor_power[act_mask], soc_rate_of_change[act_mask],
                                s=10, color=activity_color_map[act], label=act)
            
            # # Calculate and plot line of best fit for each activity
            # a, b = np.polyfit(motor_power[act_mask], soc_rate_of_change[act_mask], 1)
            # plt.plot(motor_power[act_mask], a*motor_power[act_mask] + b, 
            #                 color=activity_color_map[act], linestyle='--', linewidth=2)

        # Create a legend with unique entries
        # Handles are references to the plot elements, and labels are the text descriptions for these elements
        handles, labels = scatter_ax.get_legend_handles_labels()

        # Create a unique list of handle-label pairs
        # This is to ensure that each label (and its corresponding handle) appears only once in the legend
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]

        # Create and set the legend for the scatter plot
        # *zip(*unique) unpacks the unique handle-label pairs into separate tuples of handles and labels
        plt.legend(*zip(*unique), loc='upper right', fontsize="7")

    # Add plot stuff
    plt.xlabel("Motor Power")
    plt.ylabel("SOC Rate of Change (% change every 0.5 min)")
    plt.title("Motor Power vs. SOC Rate of Change Scatterplot")

    return scatter_figure

# Function -------------------------------------------------------------------------------------------------------------------------------------------------------
def soh_soc_rate_scatterplot(flight_ids: list):
    """
    The function takes in flight ids and dates and creates a single matplotlib figure scatter plot of SOH vs. SOC rate of change.
    A legend of dates is also included. 
    Parameters:
        flight_ids: A list of all flight ids form the DB. Index should corresponds with the flight_dates index.
        flight_dates: A list of all flight dates form the DB. Index should corresponds with the flight_ids index.
    Returns:
        scatter_figure: The matplotlib figure axis with stored scatter plot data and other supports.
    """
    
    # Set Plot
    scatter_figure = plt.figure(figsize=(6, 6))
    scatter_figure.tight_layout()

    # Plot the graphs
    for i in range(0, len(flight_ids)):
        # Make flight db connection
        print("inside for loop")
        flight_db_conn = query_flights()
        flight_data = flight_db_conn.get_flight_soh_soc_rate(flight_ids[i])
        # Define the date and id
        id = flight_ids[i]

        # Get the soh and soc and plot it with a legend label
        soh = flight_data[id]['soh']
        soc_rate_of_change = flight_data[id]['soc_rate_of_change']
        date = flight_data[id]["dates"]
        plt.scatter(soh, soc_rate_of_change, s=5, alpha = 0.3, label=date)

    plt.xlabel("SOH (%)")
    plt.ylabel("SOC Rate of Change (% change every 0.5 min)")
    plt.title("SOH vs. SOC Rate of Change Scatterplot")
    plt.legend(loc='upper right', fontsize="7", ncol=1,
            borderaxespad=0, frameon=True)

    return scatter_figure


# Function -------------------------------------------------------------------------------------------------------------------------------------------------------
def soh_plot():
    """
    The function takes in dates and soh and creates a single matplotlib figure line plot of date vs. SOH.
    Returns:
        line_figure: The matplotlib figure axis with stored line plot data and other supports.
    """
    
    # Set Plot
    line_figure = plt.figure(figsize=(6, 6))
    line_figure.tight_layout()

    # Make flight db connection
    flight_db_conn = query_flights()
    flight_data = flight_db_conn.get_flight_soh()
    soh = flight_data['soh']
    dates = flight_data["dates"]

    # Plot the graph
    plt.plot(dates, soh, marker='o', linestyle='-')
    plt.xticks(dates, rotation=45)
    plt.xlabel("Date (Year-Month)")
    plt.ylabel("SOH (%)")
    plt.title("Average SOH Per Month")

    return line_figure


# Function -------------------------------------------------------------------------------------------------------------------------------------------------------
def custom_graph_creation(graph_type: str, flight_id, x_variable: str, y_variable: str, x_label: str, y_label: str):
    """
    The function creates a graph based on the flight id, variables, and type.
    Returns:
        custome_figure: figure of a graph
    """

    # Set Plot
    custom_figure = plt.figure(figsize=(8, 8))
    custom_figure.tight_layout()

    # Make sure that all are not null, then plot.
    if flight_id != "" and graph_type != "" and y_variable != "" and x_variable != "":
    
        # Make the query connection
        flight_db_conn = query_flights()

        # Get data from x-variables
        query_result = flight_db_conn.get_flight_data_on_id(x_variable, flight_id)

        if len(x_variable) == 2:
            x_ax_data = (query_result[x_variable[0]].to_numpy() + query_result[x_variable[1]].to_numpy()) / 2
        else:
            x_ax_data = query_result[x_variable].to_numpy()

        # Get data from y-variables if they are not the same as the x-variables
        if y_variable != x_variable:
            query_result = flight_db_conn.get_flight_data_on_id(y_variable, flight_id)

        if len(y_variable) == 2:
            y_ax_data = (query_result[y_variable[0]].to_numpy() + query_result[y_variable[1]].to_numpy()) / 2
        else:
            y_ax_data = query_result[y_variable].to_numpy()
        
        if (x_variable == ['time_min']) :
            # Load Pickle File Data
            with open("Aniket_Stages\kmeans_model_with_metadata_0_waterloo.pkl", "rb") as f:
                 model_dict = pickle.load(f)

            model = model_dict["model"]
            scaler = model_dict["scaler"]
            features = model_dict["metadata"]["features_used"]
            phase_map = {0: "Phase 3", 1: "Phase 0", 2: "Phase 2", 3: "Phase 1"}
            phase_colors = {
               "Phase 0": "#9b59b6",      # Purple
               "Phase 1": "#27ae60",     # Green
               "Phase 2": "#f1c40f",    # Yellow
               "Phase 3": "#e74c3c"    # Red
            }

            # Get information for pickle data
            flight_data_for_ml = flight_db_conn.get_flight_data_on_id(["pressure_alt", "requested_torque", "motor_power", "motor_rpm", "pitch", "roll", "oat", "ias", "ground_speed", "time_min"], flight_id)
            
            # Rename to fit pickle columns
            flight_data_for_ml = flight_data_for_ml.rename(columns={
               "requested_torque":      " requested torque",
               "motor_power":           " motor power",
               "motor_rpm":             " motor rpm",
               "pitch":                 " PITCH",
               "roll":                  " ROLL",
               "oat":                   " OAT",
               "ias":                   " IAS",
               "ground_speed":          " GROUND_SPEED",
               "time_min":              " time(min)",
               "pressure_alt":          " PRESSURE_ALT"
            })

            # Add other columns that needed to be derived
            flight_data_for_ml = add_altitude(flight_data_for_ml)
            flight_data_for_ml = add_RoC(flight_data_for_ml)
            flight_data_for_ml = add_smoothed_alt(flight_data_for_ml, 15)
            flight_data_for_ml = add_smoothed_RoC(flight_data_for_ml, 15)

            # Remove nan's
            flight_data_for_ml = flight_data_for_ml.fillna(0)

            # Apply model if possible
            if all(col in flight_data_for_ml for col in features):
                 X = flight_data_for_ml[features]
                 X_scaled = scaler.transform(X)
                 cluster_labels = model.predict(X_scaled)
                 phase_names = [phase_map.get(label, label) for label in cluster_labels]
                 print("Successful_cluster_labers")
            else:
                 cluster_labels = None
                 print("failed_cluster_labels")
                 missing = [col for col in features if col not in flight_data_for_ml]
                 print("failed_cluster_labels. Missing columns:", missing)
            
            # Output graph
            if graph_type == "Line Plot":
               plt.plot(x_ax_data, y_ax_data)
               if cluster_labels is not None :
                    for i in range(2, len(x_ax_data)):
                        x_pair = [x_ax_data[i-1].item(), x_ax_data[i].item()]
                        y_pair = [y_ax_data[i-1].item(), y_ax_data[i].item()]
                        label = cluster_labels[i]
                        color = phase_colors.get(phase_map.get(label, f"Phase {label}"), "#cccccc")

                        plt.fill_between(x_pair, y_pair, 0, color=color, alpha=0.3)

                        from matplotlib.patches import Patch
                        legend_patches = [Patch(color=color, label=label) for label, color in phase_colors.items()]
                        plt.legend(handles=legend_patches, title="Flight Phase", loc="upper right")
               else:
                    plt.plot(x_ax_data, y_ax_data)

            elif graph_type == "Scatter Plot":
               if cluster_labels is not None :
                    scatter = plt.scatter(x_ax_data, y_ax_data, s=0.1, alpha=0.6, c=cluster_labels, cmap='tab10')
                    handles, _ = scatter.legend_elements(prop="colors")
                    plt.legend(handles, list(phase_map.values()), title="Flight Phase")
               else:
                    plt.scatter(x_ax_data, y_ax_data, s=0.1, alpha=0.6, c='blue')
        else : 
            # For each graph type graph different things.
            if graph_type == "Line Plot":
                 plt.plot(x_ax_data, y_ax_data)

            elif graph_type == "Scatter Plot":
                 plt.scatter(x_ax_data, y_ax_data, s=0.1, alpha = 0.6, c='blue')
        
       
        # Add labels and legend to plot
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f"{x_label} vs {y_label}")

        # Return the axis
        return custom_figure

# Function -------------------------------------------------------------------------------------------------------------------------------------------------------
def charging_graph_creation(graph_type: str, flight_ids, x_variable: str, y_variable: str, x_label: str, y_label: str):

    x_ax_data = []
    y_ax_data = []
    # Make the query connection
    flight_db_conn = query_flights()
    flight_data = flight_db_conn.get_flight_soc_and_time(flight_ids)
    print(flight_data)
    # Get data from x-variables
    for flight_id in flight_ids:
        if x_variable[0] == "temperature":
            query_result_x = flight_db_conn.get_temperature_on_id(flight_id)
        else:
            query_result_x = flight_db_conn.get_flight_data_on_id(x_variable, flight_id)

        # Get data from y-variables if they are not the same as the x-variables
        if y_variable != x_variable:
            if y_variable[0] == "temperature":
                query_result_y = flight_db_conn.get_temperature_on_id(flight_id)
            else:
                query_result_y = flight_db_conn.get_flight_data_on_id(y_variable, flight_id)

        # if one of the columns is temp, then we run the function
        if x_variable[0] == "temperature":
            query_result_x = aggregate_weather_for_charging(query_result_x, query_result_y)
        elif y_variable[0] == "temperature":
            query_result_y = aggregate_weather_for_charging(query_result_y, query_result_x)
        elif x_variable[0] and y_variable[0] == "temperature":
            query_result_y = aggregate_weather_for_charging(query_result_x, query_result_x)

        if len(x_variable) == 2:
            x_data = (query_result_x[x_variable[0]].to_numpy() + query_result_x[x_variable[1]].to_numpy()) / 2
        else:
            x_data = query_result_x[x_variable].to_numpy()

        
        if len(y_variable) == 2:
            y_data = (query_result_y[y_variable[0]].to_numpy() + query_result_y[y_variable[1]].to_numpy()) / 2
        else:
            y_data = query_result_y[y_variable].to_numpy()
        
        x_ax_data.append(x_data)
        y_ax_data.append(y_data)

    # Set Plot
    custom_figure = plt.figure(figsize=(6, 6))
    custom_figure.tight_layout()

    # For each graph type graph different things.
    for i in range(len(flight_ids)):
        x_data = x_ax_data[i]
        y_data = y_ax_data[i]
        date = flight_data[flight_ids[i]]["date"]
        if graph_type == "Line Plot":
            plt.plot(x_data, y_data, label=date)
        elif graph_type == "Scatter Plot":
            plt.scatter(x_data, y_data, s=0.1, alpha = 0.6, label=date)
    
    # Add labels and legend to plot
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{x_label} vs {y_label}")
    if graph_type == "Scatter Plot":
        handles, labels = plt.gca().get_legend_handles_labels()
        # Get colors from the scatter plot
        scatter_colors = [handle.get_facecolor()[0] for handle in handles]
        # Create custom legend with larger markers and actual scatter plot colors
        custom_handles = [plt.Line2D([0], [0], marker='o', markersize=10, linestyle='', color=color, label=label) for color, label in zip(scatter_colors, labels)]
        plt.legend(handles=custom_handles, labels=labels)
    else:
        plt.legend()

        # Get the legend handles and labels

    # Return the axis
    return custom_figure

def aggregate_weather_for_charging(weather_data, flight_data):
        
    # Extract unique values from the column in the temperature dataframe
    unique_values = weather_data['temperature'].unique()

    # Calculate the number of times each unique value should be repeated
    num_repeats = len(flight_data) // len(unique_values)
    remainder = len(flight_data) % len(unique_values)

    # Create a list of repeated values
    repeated_values = []
    for value in unique_values:
        repeated_values.extend([value] * num_repeats)

    # Append remainder of values to evenly distribute
    repeated_values.extend(unique_values[:remainder])

    # Assign the repeated values to a new column in the first dataframe
    flight_data['temperature'] = repeated_values[:len(flight_data)]

    print(flight_data.head())
    return flight_data
