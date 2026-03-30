# Data Models

Network Wrangler uses [pandera's DataFrameModel](https://pandera.readthedocs.io/en/stable/reference/generated/pandera.api.pandas.model.DataFrameModel.html) as the base class for all data validation models. The following diagrams show how the core network classes contain these data models and their inheritance relationships:

## Network Containment Diagram

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'fontSize': '12px'}}}%%
graph TD
    S[Scenario]

    S --> RN

    subgraph "RoadwayNetwork"
        RN[RoadwayNetwork]
        RL[nodes_df: RoadNodesTable]
        RK[links_df: RoadLinksTable]
        RS[shapes_df: RoadShapesTable]
        RN -.-> RL
        RN -.-> RK
        RN -.-> RS
    end

    %% Force vertical spacing
    SPACER1[" "]
    SPACER2[" "]

    RN --> SPACER1
    SPACER1 --> SPACER2
    SPACER2 --> TN

    subgraph "TransitNetwork"
        TN[TransitNetwork]
        F[Feed]
        TS[stops: WranglerStopsTable]
        TR[routes: RoutesTable]
        TT[trips: WranglerTripsTable]
        TST[stop_times: WranglerStopTimesTable]
        TSH[shapes: WranglerShapesTable]
        TF[frequencies: WranglerFrequenciesTable]
        TA[agencies: AgenciesTable]
        TN -.-> F
        F -.-> TS
        F -.-> TR
        F -.-> TT
        F -.-> TST
        F -.-> TSH
        F -.-> TF
        F -.-> TA
    end

    %% Hide spacers
    style SPACER1 fill:transparent,stroke:transparent
    style SPACER2 fill:transparent,stroke:transparent

    click S "../api/#network_wrangler.scenario"
    click RN "../api/#network_wrangler.roadway.network"
    click TN "../api/#network_wrangler.transit.network"
    click F "../api_transit/#network_wrangler.transit.feed.feed.Feed"
    click RL "../api_roadway/#network_wrangler.models.roadway.tables.RoadNodesTable"
    click RK "../api_roadway/#network_wrangler.models.roadway.tables.RoadLinksTable"
    click RS "../api_roadway/#network_wrangler.models.roadway.tables.RoadShapesTable"
    click TS "../api_transit/#network_wrangler.models.gtfs.tables.WranglerStopsTable"
    click TR "../api_transit/#network_wrangler.models.gtfs.tables.RoutesTable"
    click TT "../api_transit/#network_wrangler.models.gtfs.tables.WranglerTripsTable"
    click TST "../api_transit/#network_wrangler.models.gtfs.tables.WranglerStopTimesTable"
    click TSH "../api_transit/#network_wrangler.models.gtfs.tables.WranglerShapesTable"
    click TF "../api_transit/#network_wrangler.models.gtfs.tables.WranglerFrequenciesTable"
    click TA "../api_transit/#network_wrangler.models.gtfs.tables.AgenciesTable"

    classDef core fill:#e3f2fd,stroke:#1976d2,stroke-width:3px
    classDef container fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef roadway fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef transit fill:#fff3e0,stroke:#f57c00,stroke-width:2px

    class S,RN,TN,F core
    class RL,RK,RS roadway
    class TS,TR,TT,TST,TSH,TF,TA transit
```

### Roadway Inheritance Diagrams

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'fontSize': '12px'}}}%%
graph TD
    A["DataFrameModel"]
    A --> B["RoadLinksTable"]
    A --> C["RoadNodesTable"]
    A --> D["RoadShapesTable"]
    A --> E["ExplodedScopedLinkPropertyTable"]
    A --> F["NodeGeometryChangeTable"]

    click A "https://pandera.readthedocs.io/en/stable/reference/generated/pandera.api.pandas.model.DataFrameModel.html"
    click B "../api_roadway/#network_wrangler.models.roadway.tables.RoadLinksTable"
    click C "../api_roadway/#network_wrangler.models.roadway.tables.RoadNodesTable"
    click D "../api_roadway/#network_wrangler.models.roadway.tables.RoadShapesTable"
    click E "../api_roadway/#network_wrangler.models.roadway.tables.ExplodedScopedLinkPropertyTable"
    click F "../api_roadway/#network_wrangler.roadway.nodes.edit.NodeGeometryChangeTable"

    classDef pandera fill:#e1f5fe,stroke:#01579b,stroke-width:3px
    classDef roadway fill:#f3e5f5,stroke:#4a148c,stroke-width:2px

    class A pandera
    class B,C,D,E,F roadway
```

### Transit/GTFS Inheritance Diagrams

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'fontSize': '12px'}}}%%
graph TD
    A["DataFrameModel"]
    A --> B["AgenciesTable"]
    A --> C["StopsTable"]
    A --> D["RoutesTable"]
    A --> E["ShapesTable"]
    A --> F["TripsTable"]
    A --> G["FrequenciesTable"]
    A --> H["StopTimesTable"]

    C --> I["WranglerStopsTable"]
    E --> J["WranglerShapesTable"]
    F --> K["WranglerTripsTable"]
    G --> L["WranglerFrequenciesTable"]
    H --> M["WranglerStopTimesTable"]

    click A "https://pandera.readthedocs.io/en/stable/reference/generated/pandera.api.pandas.model.DataFrameModel.html"
    click B "../api_transit/#network_wrangler.models.gtfs.tables.AgenciesTable"
    click C "../api_transit/#network_wrangler.models.gtfs.tables.StopsTable"
    click D "../api_transit/#network_wrangler.models.gtfs.tables.RoutesTable"
    click E "../api_transit/#network_wrangler.models.gtfs.tables.ShapesTable"
    click F "../api_transit/#network_wrangler.models.gtfs.tables.TripsTable"
    click G "../api_transit/#network_wrangler.models.gtfs.tables.FrequenciesTable"
    click H "../api_transit/#network_wrangler.models.gtfs.tables.StopTimesTable"
    click I "../api_transit/#network_wrangler.models.gtfs.tables.WranglerStopsTable"
    click J "../api_transit/#network_wrangler.models.gtfs.tables.WranglerShapesTable"
    click K "../api_transit/#network_wrangler.models.gtfs.tables.WranglerTripsTable"
    click L "../api_transit/#network_wrangler.models.gtfs.tables.WranglerFrequenciesTable"
    click M "../api_transit/#network_wrangler.models.gtfs.tables.WranglerStopTimesTable"

    classDef pandera fill:#e1f5fe,stroke:#01579b,stroke-width:3px
    classDef gtfs fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef wrangler fill:#fff3e0,stroke:#e65100,stroke-width:2px

    class A pandera
    class B,C,D,E,F,G,H gtfs
    class I,J,K,L,M wrangler
```

**Legend:**

- 🔗 **DataFrameModel** - External pandera base class (links to pandera docs)
- 🟣 **Purple** - Roadway network data models  
- 🟢 **Green** - Standard GTFS transit data models
- 🟠 **Orange** - Wrangler-enhanced GTFS models with additional fields

### DBModelMixin Inheritance Diagrams

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'fontSize': '12px'}}}%%
graph TD
    A["DBModelMixin"]
    A --> B["GtfsModel"]
    A --> C["Feed"]
    A --> D["MockDBModel"]

    click A "../api/#network_wrangler.models._base.db.DBModelMixin"
    click B "../api_transit/#network_wrangler.models.gtfs.gtfs.GtfsModel"
    click C "../api_transit/#network_wrangler.transit.feed.feed.Feed"

    classDef mixin fill:#fff3e0,stroke:#e65100,stroke-width:3px
    classDef gtfs fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef wrangler fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef test fill:#f3e5f5,stroke:#4a148c,stroke-width:2px

    class A mixin
    class B gtfs
    class C wrangler
    class D test
```

**Legend:**

- 🟠 **DBModelMixin** - Base mixin for managing interrelated DataFrameModel tables
- 🟢 **GtfsModel** - Pure GTFS feed data wrapper
- 🔵 **Feed** - Wrangler-enhanced GTFS feed with additional functionality
- 🟣 **MockDBModel** - Test implementation (not shown in API docs)

💡 **Tip:** Click on any box in the diagrams to jump directly to that class's documentation!
