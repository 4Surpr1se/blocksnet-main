from geopandas import GeoDataFrame
from pydantic import BaseModel, field_validator

from blocksnet.models import PolygonGeoJSON


class NewCityModel(BaseModel):  # pylint: disable=too-many-instance-attributes,too-few-public-methods

    territory: PolygonGeoJSON | str
    roads: PolygonGeoJSON
    water: PolygonGeoJSON
    rivers: PolygonGeoJSON
    railways: PolygonGeoJSON
    buildings: PolygonGeoJSON

    def get_service_types(self) -> list[str]:
        return list(self.services.keys())

    @field_validator("territory", mode="before")
    def validate_territory(value):
        if isinstance(value, GeoDataFrame):
            return PolygonGeoJSON.from_gdf(value)
        return value

    @field_validator("roads", mode="before")
    def validate_roads(value):
        if isinstance(value, GeoDataFrame):
            return PolygonGeoJSON.from_gdf(value)
        return value

    @field_validator("water", mode="before")
    def validate_water(value):
        if isinstance(value, GeoDataFrame):
            return PolygonGeoJSON.from_gdf(value)
        return value

    @field_validator("rivers", mode="before")
    def validate_rivers(value):
        if isinstance(value, GeoDataFrame):
            return PolygonGeoJSON.from_gdf(value)
        return value

    @field_validator("railways", mode="before")
    def validate_railways(value):
        if isinstance(value, GeoDataFrame):
            return PolygonGeoJSON.from_gdf(value)
        return value

    @field_validator("buildings", mode="before")
    def validate_buildings(value):
        if isinstance(value, GeoDataFrame):
            return PolygonGeoJSON.from_gdf(value)
        return value

    @field_validator("nodev", mode="before")
    def validate_nodev(value):
        if isinstance(value, GeoDataFrame):
            return PolygonGeoJSON.from_gdf(value)
        return value
