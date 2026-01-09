# Hydrology Plot Guide - Plain English Explanations

This guide explains what each plot shows and how to interpret it. **No hydrology degree required!**

---

## Table of Contents
- [Climate-Dependent Plots](#climate-dependent-plots)
- [Discharge-Only Plots](#discharge-only-plots)
- [Understanding Plot Statistics](#understanding-plot-statistics)

---

## Climate-Dependent Plots

These plots require both streamflow and weather data (temperature and precipitation from Meteostat).

### 1. Anomaly Plot: "Are We Normal?"

**What it shows:** Compares the most recent decade to historical averages for each month.

**How to read it:**
- **Blue shaded area**: Recent decade is *cooler or drier* than historical average
- **Red shaded area**: Recent decade is *warmer or wetter* than historical average
- **Three lines shown**: Discharge (Q), Temperature (T), and Precipitation (P)

**Real-world example:**
If August shows a big red area for discharge, it means August streamflow has been higher in the last 10 years compared to the long-term average. This could indicate:
- More snowmelt in the watershed
- Changes in precipitation patterns
- Changes in water management

**Why it matters:** Helps identify climate trends and changing seasonal patterns that affect water availability.

---

### 2. Hexbin Temperature Plot: "Does Heat = Low Water?"

**What it shows:** Relationship between temperature and streamflow, with colors showing how often that combination occurs.

**How to read it:**
- **X-axis**: Average daily temperature (°C)
- **Y-axis**: Streamflow (cubic feet per second, log scale)
- **Colors**:
  - **Purple/Blue**: Rare combinations (not common)
  - **Green/Yellow**: Common combinations (happens often)

**Real-world example:**
You'll typically see:
- **Cold temps (0-10°C)** + **High flow**: Spring snowmelt
- **Hot temps (20-30°C)** + **Low flow**: Summer drought
- **Yellow/green band**: The most common temp-flow relationship for this river

**Why it matters:** Shows if your river is temperature-sensitive (e.g., snowmelt-driven vs rain-driven).

---

### 3. Lagged Precipitation Plot: "Does Yesterday's Rain Matter?"

**What it shows:** How streamflow today relates to precipitation from 1 day ago, broken down by month.

**How to read it:**
- **X-axis**: How much it rained yesterday (mm)
- **Y-axis**: How much water is flowing today (cfs)
- **Colors**: Each month has a different color (see legend)
- **Diagonal trend**: If points go up-right, yesterday's rain affects today's flow

**Real-world example:**
- **Steep upward trend**: Fast response - rain quickly reaches the river (urban area or thin soil)
- **Flat/scattered**: Slow response - water takes days to reach river (forest, wetlands, or deep aquifer)

**Statistical note:**
- **r = correlation coefficient** (-1 to +1)
  - r > 0.5: Strong positive relationship
  - r near 0: No relationship
  - r < -0.5: Strong negative relationship (rare for precip vs flow)
- **p-value**: If p < 0.05, the relationship is statistically significant (not just random chance)

**Why it matters:** Helps predict flood response time and understand watershed characteristics.

---

### 4. Correlation Matrix: "What Affects What?"

**What it shows:** How strongly discharge, temperature, and precipitation are related to each other.

**How to read the heatmap:**
- **Numbers (-1.00 to +1.00)**: Correlation strength
  - **+1.00**: Perfect positive relationship (when one goes up, other goes up)
  - **0.00**: No relationship (completely independent)
  - **-1.00**: Perfect negative relationship (when one goes up, other goes down)

- **Colors**:
  - **Deep Red**: Strong positive correlation (+0.8 to +1.0)
  - **Light Pink**: Moderate positive correlation (+0.4 to +0.8)
  - **White**: No correlation (near 0)
  - **Light Blue**: Moderate negative correlation (-0.4 to -0.8)
  - **Deep Blue**: Strong negative correlation (-0.8 to -1.0)

- **Stars**: Statistical significance
  - **★★★**: p < 0.001 (extremely confident this relationship is real)
  - **★★**: p < 0.01 (very confident)
  - **★**: p < 0.05 (confident)
  - **ns**: Not significant (could be random chance)

**Real-world example:**

| Relationship | Typical Value | What It Means |
|--------------|---------------|---------------|
| **Discharge vs Temp** | -0.3 to -0.6 ★★ | Hotter = less water (summer drought, evaporation) |
| **Discharge vs Precip** | +0.2 to +0.5 ★ | More rain = more water (but often delayed) |
| **Temp vs Precip** | -0.1 to +0.1 ns | Temperature and rain often unrelated in same period |

**Why it matters:**
- Negative discharge-temperature correlation? Your river is **temperature-sensitive** (snowmelt or high evaporation).
- Strong discharge-precipitation correlation? Your river is **rain-dominated** (quick runoff).
- Weak correlations? Complex system with **multiple water sources** (groundwater, snowmelt, rain).

---

### 5. Precipitation-Discharge Overlay: "Rain vs River Response"

**What it shows:** Daily precipitation bars overlaid with streamflow line, plus cumulative precipitation.

**How to read it:**
- **Blue line (left axis)**: Daily streamflow (cfs)
- **Light blue bars (right axis)**: Daily precipitation (mm)
- **Orange line (right axis)**: Cumulative precipitation (adds up over time)

**Real-world patterns:**
1. **Immediate response**: Stream jumps right after rain bars → Urban watershed or thin soil
2. **Delayed response**: Stream peaks days after rain → Forested watershed with groundwater buffering
3. **No response**: Rain but no flow increase → Dry soil soaking up water, or snow accumulation
4. **Flow without rain**: Groundwater discharge, snowmelt, or upstream releases

**Why it matters:**
- Helps predict **flood timing** after storms
- Shows **drought recovery** (how much rain needed to restore flow)
- Identifies **dry season dependence** on groundwater

---

## Discharge-Only Plots

These plots only need streamflow data - they work even if weather data is unavailable.

### 6. Timeseries: "What's Happening Recently?"

**What it shows:** Daily streamflow over the most recent period (usually 3-5 years).

**How to read it:**
- **X-axis**: Date
- **Y-axis**: Streamflow (cubic feet per second, log scale)
- **Peaks**: High flow events (floods, snowmelt)
- **Valleys**: Low flow events (droughts, summer base flow)

**Seasonal patterns:**
- **Spring peaks**: Snowmelt (mountain rivers) or spring rains
- **Summer lows**: Drought, irrigation withdrawals, high evaporation
- **Fall recovery**: Rain returns, irrigation demand drops
- **Winter**: Depends on climate (rain in wet climates, frozen in cold climates)

**Why it matters:** Shows current conditions and recent extreme events (floods, droughts).

---

### 7. Flow Duration Curve: "How Often is the River This High?"

**What it shows:** What percentage of time streamflow is at or above a certain level.

**How to read it:**
- **X-axis**: Exceedance Probability (%)
  - **0%**: Highest flow ever recorded (extreme flood)
  - **50%**: Median flow (half the time it's higher, half the time it's lower)
  - **100%**: Lowest flow ever recorded (extreme drought)
- **Y-axis**: Streamflow (cfs, log scale)

**Key percentiles:**
- **Q10 (10% exceedance)**: High flow - exceeded only 10% of the time (flood threshold)
- **Q50 (50% exceedance)**: Median flow - typical "normal" flow
- **Q90 (90% exceedance)**: Low flow - only 10% of the time is it lower (drought threshold)

**Curve shape tells a story:**
- **Steep curve**: Flashy river with big swings (urban, steep watershed)
- **Flat curve**: Stable river with consistent flow (groundwater-fed, large lakes)

**Real-world use:**
- **Water rights**: "You can withdraw water when flow exceeds Q90"
- **Habitat**: "Fish need at least Q95 to survive"
- **Recreation**: "Rafting requires flows above Q30"

**Why it matters:** Critical for water management, environmental flows, and infrastructure design.

---

### 8. Monthly Boxplot: "What's Normal for Each Month?"

**What it shows:** Distribution of streamflow for each month across all years of data.

**How to read a boxplot:**
```
    ╷ ← Maximum (excluding outliers)
    ┤
┌───┐
│   │ ← Box: 75th percentile (top), 25th percentile (bottom)
├───┤ ← Median (middle line in box)
│   │
└───┘
    ┤
    ╵ ← Minimum (excluding outliers)
    •   ← Outliers (extreme years)
```

**Real-world example:**
- **January**: High median with large box → Variable winter flows
- **July**: Low median with small box → Consistently dry
- **May**: High median with outliers above → Occasional flood years

**Why it matters:**
- Shows **seasonal reliability** (small box = predictable, large box = variable)
- Identifies **flood/drought months**
- Helps plan **reservoir operations** and **irrigation schedules**

---

### 9. Discharge Heatmap: "When Does High/Low Flow Typically Occur?"

**What it shows:** A 2D histogram showing discharge patterns across the entire year.

**How to read it:**
- **X-axis**: Day of year (1 = Jan 1, 365 = Dec 31)
- **Y-axis**: Streamflow (cfs, log scale)
- **Colors**:
  - **Purple/Blue**: Rare (this flow almost never happens on this day)
  - **Green/Yellow**: Common (this flow typically happens on this day)

**Real-world patterns:**
- **Yellow band in April-June at high flows**: Predictable spring snowmelt
- **Yellow band in July-September at low flows**: Predictable summer drought
- **Scattered purple everywhere**: Unpredictable, variable river

**Why it matters:** Shows seasonal flow patterns and helps identify the "typical" hydrology for any day of the year.

---

### 10. Temporal Heatmap: "Is the River Changing Over Time?"

**What it shows:** Four panels comparing discharge patterns across different time periods.

**Four panels explained:**
1. **Last 5 Years**: Most recent - shows current conditions
2. **Last 10 Years**: Medium-term - captures recent changes
3. **Last 20 Years**: Long-term - includes climate variability
4. **Total Record**: All available data - the "big picture"

**How to compare panels:**
- **Same patterns in all 4**: Stable hydrology (no significant changes)
- **Recent panels different**: Changing conditions (climate, land use, water management)
- **Look at yellow bands**: Are high flow days shifting earlier/later in the year?

**Real-world insights:**
- **Spring peak shifting earlier**: Warmer climate, earlier snowmelt
- **Summer low flows getting lower**: Increased irrigation, less snowpack
- **More scatter in recent panels**: Increasing climate variability

**Why it matters:** Detects long-term trends in river behavior that affect water planning and ecosystem health.

---

## Understanding Plot Statistics

### Correlation Coefficient (r)

**What it measures:** How strongly two things move together.

**Scale:**
- **r = +1.0**: Perfect positive relationship (always move together)
- **r = +0.7 to +1.0**: Strong positive relationship
- **r = +0.4 to +0.7**: Moderate positive relationship
- **r = +0.1 to +0.4**: Weak positive relationship
- **r = 0**: No relationship at all
- **r = -0.1 to -0.4**: Weak negative relationship (move opposite)
- **r = -0.4 to -0.7**: Moderate negative relationship
- **r = -0.7 to -1.0**: Strong negative relationship
- **r = -1.0**: Perfect negative relationship (always move opposite)

**Example in hydrology:**
- **Discharge vs Precipitation: r = +0.5**: When it rains more, the river tends to be higher, but not always (groundwater delay, seasonality)
- **Discharge vs Temperature: r = -0.4**: When it's hotter, the river tends to be lower (evaporation, irrigation), but weakly

### P-Value (Statistical Significance)

**What it measures:** How confident are we this relationship isn't just random luck?

**Scale:**
- **p < 0.001 (★★★)**: 99.9% confident this is real (extremely significant)
- **p < 0.01 (★★)**: 99% confident this is real (very significant)
- **p < 0.05 (★)**: 95% confident this is real (significant)
- **p > 0.05 (ns)**: Not significant - could be random chance

**Real-world interpretation:**
- **r = 0.8, p = 0.001**: Strong relationship that's definitely real (trust this!)
- **r = 0.8, p = 0.10**: Strong relationship but might be random (not enough data)
- **r = 0.2, p = 0.001**: Weak relationship but definitely exists (large dataset)
- **r = 0.2, p = 0.50**: Weak relationship that's probably just noise (ignore this)

**Rule of thumb:** Only trust relationships where **both** r is strong AND p is significant.

---

## Quick Reference: Plot Selection Guide

| Question | Recommended Plot |
|----------|------------------|
| Is this river changing with climate? | Anomaly, Correlation Matrix, Temporal Heatmap |
| When does flooding typically happen? | Monthly Boxplot, Discharge Heatmap, Timeseries |
| How reliable is this water source? | Flow Duration Curve, Monthly Boxplot |
| Does rain quickly reach the river? | Lagged Precip, Precip-Discharge Overlay |
| Is this river temperature-sensitive? | Hexbin Temp, Correlation Matrix, Anomaly |
| What's normal vs extreme flow? | Flow Duration Curve, Monthly Boxplot |
| How much does flow vary seasonally? | Monthly Boxplot, Discharge Heatmap, Timeseries |
| Is the river getting flashier/more variable? | Temporal Heatmap, Monthly Boxplot comparison |

---

## Tips for Interpretation

### 1. Context Matters
- **Mountain river**: Expect strong snowmelt signal, negative temp correlation
- **Prairie river**: Expect rain-dominated, positive precip correlation
- **Groundwater-fed river**: Expect stable flow, weak weather correlations
- **Urban river**: Expect flashy response, quick precip correlation

### 2. Look for Patterns Across Multiple Plots
Don't rely on a single plot! For example:
- **Anomaly** shows discharge increasing in spring
- **Temporal heatmap** shows spring peak shifting earlier
- **Correlation matrix** shows negative discharge-temp correlation
- **Conclusion**: Earlier, warmer springs = earlier snowmelt

### 3. Data Quality Flags
- **Lots of "ns" (not significant)**: Small dataset or very noisy data
- **Perfect correlations (r = 1.0)**: Suspicious - check for data errors
- **Empty plots or "Data N/A"**: Missing climate data (check Meteostat availability)

### 4. Seasonal vs Annual Patterns
- Monthly boxplots show **within-year** patterns (spring vs summer)
- Temporal heatmaps show **across-years** patterns (2000s vs 2020s)
- Use both to distinguish weather (short-term) from climate (long-term)

---

## Further Resources

- **USGS Water Data**: https://waterdata.usgs.gov/nwis
- **Meteostat Climate Data**: https://meteostat.net
- **Understanding Flow Duration Curves**: USGS educational resources
- **Hydrology Basics**: USGS Water Science School

---

## Contributing to This Guide

Found an unclear explanation? Have a better real-world example? Please submit issues or pull requests to improve this guide for everyone!
