from dataclasses import dataclass
from typing import Dict, List, Optional

import folium
import streamlit as st

from streamlit_folium import st_folium




@dataclass
class Point:
    lat: float
    lon: float

    @classmethod
    def from_dict(cls, data: Dict) -> "Point":
        if "lat" in data:
            return cls(float(data["lat"]), float(data["lng"]))
        elif "latitude" in data:
            return cls(float(data["latitude"]), float(data["longitude"]))
        else:
            raise NotImplementedError(data.keys())

    def is_close_to(self, other: "Point") -> bool:
        close_lat = self.lat - 0.0001 <= other.lat <= self.lat + 0.0001
        close_lon = self.lon - 0.0001 <= other.lon <= self.lon + 0.0001
        return close_lat and close_lon


@dataclass
class Bounds:
    south_west: Point
    north_east: Point

    def contains_point(self, point: Point) -> bool:
        in_lon = self.south_west.lon <= point.lon <= self.north_east.lon
        in_lat = self.south_west.lat <= point.lat <= self.north_east.lat

        return in_lon and in_lat

    @classmethod
    def from_dict(cls, data: Dict) -> "Bounds":
        return cls(
            Point.from_dict(data["_southWest"]), Point.from_dict(data["_northEast"])
        )


#############################
# Streamlit app
#############################

"## National Parks in the United States"

"""
The National Parks Service provides an
[API](https://www.nps.gov/subjects/digital/nps-data-api.htm) to programmatically explore
NPS data.

We can take data about each park and display it on the map _conditionally_ based on
whether it is in the viewport.

---
"""

def maps_folim():
    # define layout
    c1, c2 = st.columns(2)

    # get and cache data from API
    parks = pd.read_parquet('final_conus_v2.parquet')

    # layout map
    with c1:
        """(_Click on a pin to bring up more information_)"""
        m = folium.Map(location=[42.0285, -93.85, zoom_start=4)

        for park in parks:
            popup = folium.Popup(
                f"""
                      <a href="{park["soil_organic_carbon"]}" target="_blank">{park["soil_organic_carbon"]}</a><br>
                      <br>
                      {park["soil_organic_carbon"]}<br>
                      <br>
                      Soil organic carbon: {park["soil_organic_carbon"]}<br>
                      """,
                max_width=250,
            )
            folium.Marker([park["latitude"], park["longitude"]], popup=popup).add_to(m)

        map_data = st_folium(m, key="fig1", width=700, height=700)

    # get data from map for further processing
    map_bounds = Bounds.from_dict(map_data["bounds"])

    # when a point is clicked, display additional information about the park
    try:
        point_clicked: Optional[Point] = Point.from_dict(map_data["last_object_clicked"])

        if point_clicked is not None:
            with st.spinner(text="loading image..."):
                for park in parks:
                    if park["_point"].is_close_to(point_clicked):
                        with c2:
                            f"""### _{park["fullName"]}_"""
                            park["soil_organic_carbon"]
                            st.image(
                                park["soil_organic_carbon"],
                                caption=park["soil_organic_carbon"],
                            )
                            st.expander("Show park full details").write(park)
    except TypeError:
        point_clicked = None

    # even though there is a c1 reference above, we can do it again
    # output will get appended after original content
    with c1:
        parks_in_view: List[Dict] = []
        for park in parks:
            if map_bounds.contains_point(park["_point"]):
                parks_in_view.append(park)

        "Parks visible:", len(parks_in_view)
        "Bounding box:", map_bounds