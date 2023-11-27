from geopandas import GeoDataFrame, read_file
from pydantic import BaseModel, field_validator

from blocksnet.models import PolygonGeoJSON
from blocksnet.models.new_module import InfrastructureCityModel, fill_holes


class InfrastructureCityModelValidate(InfrastructureCityModel):  # pylint: disable=too-many-instance-attributes,too-few-public-methods

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


class InfrastructureCityModelGenerator(InfrastructureCityModelValidate):

    @staticmethod
    def get_bounding_polygon(gdf:GeoDataFrame, crs=4326):
        all_polygons_gdf = gdf.make_valid().unary_union
        all_polygons_gdf = fill_holes(GeoDataFrame(geometry=[all_polygons_gdf], crs=crs))
        all_polygons_gdf['area'] = all_polygons_gdf.to_crs(3857).area
        bounding_polygon = all_polygons_gdf.sort_values('area', ascending=False).iloc[0]['geometry']
        return bounding_polygon

    def explore(self, nodev: GeoDataFrame, pzz: GeoDataFrame, *args, **kwargs):
        self.generate_blocks()
        self.cut_nodev(nodev)
        self.add_pzz(pzz)
        self.blocks.explore()


# pzz = read_file('pzz_2019.geojson').to_crs(4326)
# nodev = read_file('no_development_pzz.geojson')
# ter = InfrastructureCityModelGenerator.get_bounding_polygon(pzz)
# spb = InfrastructureCityModelGenerator(territory=ter, territory_name='Санкт-Петербург')
# spb.explore(nodev, pzz)
