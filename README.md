## PROBLEM STATEMENT - 2
# AI-Driven Dynamic Public Transportation Scheduler

Efficiently managing public transportation in urban areas is a significant challenge, especially with fluctuating commuter demand, traffic variations, and unexpected events. This project aims to develop an AI-driven platform that autonomously schedules and dispatches public transport vehicles, ensuring dynamic adaptability and optimized efficiency.

## Problem Statement

As urban populations grow, managing public transportation to meet dynamic demand becomes more complex. Traditional scheduling systems often struggle to adapt to real-time fluctuations in commuter demand, traffic conditions, or unscheduled events like concerts, sports matches, and road closures. This can result in:

- Overcrowded buses or trains.
- Under-utilized vehicles.
- Longer commute times and delays.

These inefficiencies impact both commuters and transportation authorities, necessitating an intelligent, adaptive system for real-time scheduling and routing.

## Solution Overview

### Deliverables

Develop an AI-based platform that autonomously manages the scheduling, routing, and dispatching of public transportation vehicles (buses, trains, etc.) based on real-time data and predictive insights. The system should:

- **Predict Commuter Demand:** Leverage historical and live data to forecast demand across routes.
- **Adapt Scheduling in Real-Time:** Respond to live traffic conditions and events by dynamically adjusting schedules.
- **Optimize Routing and Dispatching:** Minimize congestion, reduce wait times, and balance vehicle utilization.

### Objectives

1. **Real-Time Commuter Demand Prediction**  
   - Use AI/ML algorithms to predict commuter demand based on historical patterns, seasonal trends, and real-time data (e.g., weather, local events, traffic data).
   
2. **Dynamic Scheduling and Routing**  
   - Continuously adapt transport schedules and routes in response to live traffic conditions and known events (e.g., concerts, sporting events, road closures).
   
3. **Optimization of Dispatching**  
   - Maximize vehicle utilization by balancing commuter loads, reducing underutilized runs, and minimizing commuter wait times.
   - Reduce congestion and ensure equitable coverage across urban regions.

## Key Features

- **Real-Time Data Integration:** Incorporates traffic data, commuter density, and event information to make data-driven decisions.
- **Predictive Analytics for Demand Forecasting:** Utilizes machine learning algorithms for accurate demand predictions.
- **Adaptive Scheduling & Routing:** Uses AI algorithms to adjust vehicle deployment and routes dynamically.
- **Event Awareness:** Automatically adjusts based on both scheduled (e.g., concerts) and unexpected events (e.g., road closures).
- **Optimization Engine:** Ensures optimal vehicle dispatching to balance demand and minimize commuter wait times.
## Program
```
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

metro_frequency_normal = 5
mrt_frequency_normal = 10
bus_frequency_normal = 15
metro_frequency_peak = 3
mrt_frequency_peak = 5
bus_frequency_peak = 10
metro_threshold = 1000
mrt_threshold = 500
bus_threshold = 300
metro_capacity = 200
mrt_capacity = 150
bus_capacity = 50

def generate_synthetic_data(num_samples=100):
    data = []
    for _ in range(num_samples):
        traffic = random.randint(1, 5)
        crowd_size = random.randint(100, 1500)
        event_time = random.randint(18, 22)
        metro_demand_1 = min(crowd_size, 200 * random.randint(1, 2))
        mrt_demand_1 = min(crowd_size, 150 * random.randint(1, 2))
        bus_demand_1 = min(crowd_size, 50 * random.randint(1, 2))
        metro_demand_2 = metro_demand_1 * 2
        mrt_demand_2 = mrt_demand_1 * 2
        bus_demand_2 = bus_demand_1 * 2
        metro_demand_3 = metro_demand_1 * 1.5
        mrt_demand_3 = mrt_demand_1 * 1.5
        bus_demand_3 = bus_demand_1 * 1.5
        metro_demand_4 = metro_demand_1 * 2.5
        mrt_demand_4 = mrt_demand_1 * 2.5
        bus_demand_4 = bus_demand_1 * 2.5
        data.append([
            traffic, crowd_size, event_time, 
            metro_demand_1, mrt_demand_1, bus_demand_1,
            metro_demand_2, mrt_demand_2, bus_demand_2,
            metro_demand_3, mrt_demand_3, bus_demand_3,
            metro_demand_4, mrt_demand_4, bus_demand_4
        ])
    df = pd.DataFrame(data, columns=[
        'traffic', 'crowd_size', 'event_time', 
        'metro_demand_1', 'mrt_demand_1', 'bus_demand_1',
        'metro_demand_2', 'mrt_demand_2', 'bus_demand_2',
        'metro_demand_3', 'mrt_demand_3', 'bus_demand_3',
        'metro_demand_4', 'mrt_demand_4', 'bus_demand_4'
    ])
    return df

df = generate_synthetic_data()
encoder = LabelEncoder()
df['event_time'] = encoder.fit_transform(df['event_time'])

X = df[['traffic', 'crowd_size', 'event_time']]
y_metro_1 = df['metro_demand_1']
y_mrt_1 = df['mrt_demand_1']
y_bus_1 = df['bus_demand_1']
y_metro_2 = df['metro_demand_2']
y_mrt_2 = df['mrt_demand_2']
y_bus_2 = df['bus_demand_2']

X_train, X_test, y_train_metro_1, y_test_metro_1 = train_test_split(X, y_metro_1, test_size=0.2, random_state=42)
X_train, X_test, y_train_mrt_1, y_test_mrt_1 = train_test_split(X, y_mrt_1, test_size=0.2, random_state=42)
X_train, X_test, y_train_bus_1, y_test_bus_1 = train_test_split(X, y_bus_1, test_size=0.2, random_state=42)
X_train, X_test, y_train_metro_2, y_test_metro_2 = train_test_split(X, y_metro_2, test_size=0.2, random_state=42)
X_train, X_test, y_train_mrt_2, y_test_mrt_2 = train_test_split(X, y_mrt_2, test_size=0.2, random_state=42)
X_train, X_test, y_train_bus_2, y_test_bus_2 = train_test_split(X, y_bus_2, test_size=0.2, random_state=42)

model_metro_1 = RandomForestRegressor(n_estimators=100, random_state=42)
model_metro_1.fit(X_train, y_train_metro_1)
model_mrt_1 = RandomForestRegressor(n_estimators=100, random_state=42)
model_mrt_1.fit(X_train, y_train_mrt_1)
model_bus_1 = RandomForestRegressor(n_estimators=100, random_state=42)
model_bus_1.fit(X_train, y_train_bus_1)
model_metro_2 = RandomForestRegressor(n_estimators=100, random_state=42)
model_metro_2.fit(X_train, y_train_metro_2)
model_mrt_2 = RandomForestRegressor(n_estimators=100, random_state=42)
model_mrt_2.fit(X_train, y_train_mrt_2)
model_bus_2 = RandomForestRegressor(n_estimators=100, random_state=42)
model_bus_2.fit(X_train, y_train_bus_2)

y_pred_metro_1 = model_metro_1.predict(X_test)
y_pred_mrt_1 = model_mrt_1.predict(X_test)
y_pred_bus_1 = model_bus_1.predict(X_test)
y_pred_metro_2 = model_metro_2.predict(X_test)
y_pred_mrt_2 = model_mrt_2.predict(X_test)
y_pred_bus_2 = model_bus_2.predict(X_test)

def adjust_frequency(demand, mode, is_peak_time=False):
    if demand > metro_threshold and mode == 'metro':
        return max(metro_frequency_peak, metro_frequency_normal - 1)
    elif demand > mrt_threshold and mode == 'mrt':
        return max(mrt_frequency_peak, mrt_frequency_normal - 1)
    elif demand > bus_threshold and mode == 'bus':
        return max(bus_frequency_peak, bus_frequency_normal - 1)
    else:
        if mode == 'metro':
            return metro_frequency_normal
        elif mode == 'mrt':
            return mrt_frequency_normal
        elif mode == 'bus':
            return bus_frequency_normal

def display_transport_status_and_recommendation(traffic, crowd_size, event_time, is_peak_time, route):
    if route == "Route 1":
        metro_demand = y_pred_metro_1[0]
        mrt_demand = y_pred_mrt_1[0]
        bus_demand = y_pred_bus_1[0]
    else:
        metro_demand = y_pred_metro_2[0]
        mrt_demand = y_pred_mrt_2[0]
        bus_demand = y_pred_bus_2[0]
    metro_freq = adjust_frequency(metro_demand, 'metro', is_peak_time)
    mrt_freq = adjust_frequency(mrt_demand, 'mrt', is_peak_time)
    bus_freq = adjust_frequency(bus_demand, 'bus', is_peak_time)
    passengers_handled_per_hour_metro = (60 / metro_freq) * metro_capacity
    passengers_handled_per_hour_mrt = (60 / mrt_freq) * mrt_capacity
    passengers_handled_per_hour_bus = (60 / bus_freq) * bus_capacity
    print(f"\nTransport Status for {route} during {'peak' if is_peak_time else 'normal'} time:")
    print(f"\nMetro Status:")
    print(f"  - Frequency: Every {metro_freq} minutes")
    print(f"  - Estimated passengers: {metro_demand}")
    print(f"  - Capacity per vehicle: {metro_capacity}")
    print(f"  - Passengers handled per hour: {passengers_handled_per_hour_metro}")
    print(f"\nMRT Status:")
    print(f"  - Frequency: Every {mrt_freq} minutes")
    print(f"  - Estimated passengers: {mrt_demand}")
    print(f"  - Capacity per vehicle: {mrt_capacity}")
    print(f"  - Passengers handled per hour: {passengers_handled_per_hour_mrt}")
    print(f"\nBus Status:")
    print(f"  - Frequency: Every {bus_freq} minutes")
    print(f"  - Estimated passengers: {bus_demand}")
    print(f"  - Capacity per vehicle: {bus_capacity}")
    print(f"  - Passengers handled per hour: {passengers_handled_per_hour_bus}")
    max_capacity = max(passengers_handled_per_hour_metro, passengers_handled_per_hour_mrt, passengers_handled_per_hour_bus)
    recommended_mode = 'Metro' if max_capacity == passengers_handled_per_hour_metro else 'MRT' if max_capacity == passengers_handled_per_hour_mrt else 'Bus'
    print(f"\nRecommended Transport Mode for {route}: {recommended_mode}")

display_transport_status_and_recommendation(
    traffic=3,
    crowd_size=1200,
    event_time=20,
    is_peak_time=True,
    route="Route 1"
)

```
---
### Note: This serves only as a reference example. Innovative ideas and unique implementation techniques are highly encouraged and warmly welcomed!
