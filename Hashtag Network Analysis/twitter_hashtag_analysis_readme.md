
# COVID-19 Twitter Hashtag Misinformation Analysis

## Project Overview

This project analyzes the network of hashtags used in tweets related to COVID-19, with a focus on identifying patterns between hashtags associated with misinformation and those associated with reliable information. Using network analysis and visualization techniques, we mapped relationships between hashtags to understand how misinformation spreads through Twitter conversations.

## Dataset

The analysis used a merged dataset containing tweets related to COVID-19 with two primary columns:

-   **Content**: The text of the tweet
-   **Label**: Classification as either "Misinformation" or "Reliable"

## Methodology

1.  **Hashtag Extraction**: Used regex to extract hashtags from tweet content
2.  **Network Creation**: Built a graph where:
    -   Nodes represent individual hashtags
    -   Edges represent co-occurrence of hashtags within the same tweet
    -   Edge weight indicates frequency of co-occurrence
3.  **Centrality Calculations**:
    -   Degree centrality: measures popularity of hashtags
    -   Betweenness centrality: identifies hashtags that bridge different topics
    -   Eigenvector centrality: identifies influential hashtags
4.  **Misinformation Analysis**:
    -   Calculated a "Misinformation Ratio" for each hashtag (proportion of occurrences in tweets labeled as misinformation)
    -   Visualized the network with colors representing this ratio (red = high misinformation, blue = reliable)

## Key Findings

### Misinformation Clusters

The visualization reveals distinct clusters of hashtags with high misinformation ratios (red nodes):

1.  **Political Cluster**: Hashtags like `#donaldtrump` show high misinformation ratios, suggesting politicization of COVID-19 information
2.  **Location-Based Misinformation**: Cities in Asia (`#wuhan`, `#beijing`, `#shanghai`, `#taipei`, `#hongkong`) show higher misinformation ratios
3.  **Travel & Tourism**: Tags like `#tourism`, `#travel`, `#cruises`, `#passengers`, `#airports` form a cluster with elevated misinformation, likely related to early pandemic travel concerns
4.  **General Virus Information**: Tags like `#virus`, `#bacteria` have higher misinformation ratios than more specific medical terms

### Reliable Information Clusters

Hashtags with lower misinformation ratios (blue nodes) form several distinct groups:

1.  **Public Health Messaging**: Tags like `#stayhome`, `#socialdistancing`, `#wearamask`, `#washyourhands` show low misinformation rates
2.  **Medical/Scientific Terms**: `#publichealth`, `#healthcare`, `#flattenthecurve` have lower misinformation ratios
3.  **Support & Awareness**: Tags like `#strongertogether`, `#afmawareness` appear more in reliable content

### Bridging Hashtags

Some hashtags serve as bridges between misinformation and reliable information clusters:

1.  **Core COVID Terms**: `#covid19`, `#coronavirus`, `#corona` appear frequently in both reliable and misinformation content
2.  **Geographic Identifiers**: `#india`, `#usa`, `#gujarat` connect different information communities

### Centrality Analysis

-   **Highest Degree Centrality**: `#covid19`, `#coronavirus` (expected as the core topics)
-   **High Betweenness Centrality**: Tags that connect different topic clusters, like `#china` and `#health`
-   **Influential Hashtags**: Tags with high eigenvector centrality shape the conversation more broadly

## Implications

1.  **Information Campaigns**: Public health organizations should focus on incorporating reliable hashtags in their messaging while monitoring and potentially countering hashtags with high misinformation ratios
    
2.  **Early Warning System**: Monitoring shifts in the misinformation ratio of bridge hashtags could provide early warning of misinformation campaigns
    
3.  **Geographic Targeting**: Location-specific misinformation patterns suggest the need for targeted fact-checking and information campaigns in specific regions
    
4.  **Topic-Specific Education**: Areas with high misinformation (travel, economic impact) could benefit from focused educational campaigns
    

## Technical Implementation

The analysis was implemented using:

-   Python for data processing
-   NetworkX for graph analysis
-   Matplotlib and Seaborn for visualization
-   Pandas for data manipulation

The code extracts hashtags, builds the network, calculates centrality measures, and visualizes the relationships with node size representing degree centrality and color representing the misinformation ratio.


