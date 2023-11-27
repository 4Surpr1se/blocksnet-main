# -*- coding: utf-8 -*-

from typing import Literal, Any

import pandas as pd
import geopandas as gpd
import numpy as np
import momepy as mm
import osmnx as ox
from blocksnet.models import CityModel
from dask.dataframe import Series
from geopandas import GeoDataFrame
from numpy import ndarray
from osmnx.features import InsufficientResponseError
import shapely
from pandas import DataFrame
from pandas._typing import NDFrameT
from pydantic import *
from shapely import (
    Point,
    MultiPoint,
    Polygon,
    MultiPolygon,
    LineString,
    MultiLineString,
)
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MiniBatchKMeans
import dask_geopandas

import pyproj
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_crs_info
from pyproj import CRS

from tqdm.notebook import tqdm
tqdm.pandas()

#service_tags = pd.read_json("service_tags.json")
service_tags = {"leisure":{"amenity":["cinema","public_bath","bbq","nightclub","casino"],"leisure":["bird_hide","picnic_table","dog_park","sauna","fitness_station","swimming_pool","water_park","resort"],"tourism":["zoo","aquarium","theme_park"]},"nature": {"leisure": ["park", "beach_resort"], "landuse": ["meadow"], "natural": ["forest", "beach"]},"education": {"amenity": ["childcare", "school", "university", "kindergarten", "college", "language_school", "driving_school", "music_school", "dancing_school", "prep_school"]},"car": {"amenity": ["fuel","charging_station","car_wash","vehicle_inspection"],"shop":["car","car_repair","car_parts","tyres"]},"accomodation": {"tourism": ["alpine_hut", "caravan_site", "motel", "hostel", "wilderness_hut", "camp_site", "chalet", "apartment", "hotel", "camp_pitch", "guest_house"]}, "culture": {"tourism": ["museum", "galery", "artwork"], "amenity": ["theatre", "library", "public_bookcase", "planetarium", "arts_centre", "studio"]}, "craft": {"craft": ["basket_maker", "sun_protection", "print_shop", "sailmaker", "carpet_layer", "tailor", "watchmaker", "plasterer", "sweep", "beekeeper", "winery", "key_cutter", "clockmaker", "sculptor", "metal_construction", "plumber", "cabinet_maker", "upholsterer", "scaffolder", "photo_studio", "tiler", "boatbuilder", "mason", "heating_engineer", "brewery", "caterer", "cleaning", "stand_builder", "glass", "painter", "confectionery", "distillery", "furniture", "photographer", "hvac", "pottery", "leather", "window_construction", "saddler", "floorer", "shoemaker", "handicraft", "builder", "roofer", "carpenter", "signmaker", "stonemason", "sawmill", "bookbinder", "dressmaker", "glaziery", "rigger", "tinsmith", "joiner", "blacksmith", "metal_works", "laboratory", "insulation", "gardener", "electrician", "turner", "jeweller", "parquet_layer", "optician", "locksmith", "electronics_repair", "agricultural_engines", "photographic_laboratory"]}, "food": {"amenity": ["restaurant", "fast_food", "fast_food", "food_court", "cafe", "bubble_tea", "ice_cream", "pub", "biergarten", "bar"]}, "office": {"office": ["charity", "foundation", "advertising_agency", "energy_supplier", "newspaper", "telecommunication", "estate_agent", "employment_agency", "company", "notary", "tax_advisor", "logistics", "travel_agent", "financial", "diplomatic", "architect", "lawyer", "water_utility", "cooperative", "association", "forestry", "camping", "consulting", "administrative", "religion", "guide", "office", "educational_institution", "accountant", "political_party", "therapist", "research", "government", "it", "vacant", "physician", "parish", "surveyor", "insurance", "quango", "ngo", "publisher", "moving_company", "register"]}, "shop": {"amenity": ["vending_machine", "marketplace"], "beauty": ["nails"], "craft": ["furniture", "confectionery", "pottery", "leather", "tailor", "optician", "locksmith"], "office": ["charity", "religion"], "shop": ["organic", "outdoor", "cheese", "hearing_aids", "interior_decoration", "anime", "window_blind", "scuba_diving", "deli", "video", "greengrocer", "bakery", "sports", "shoe_repair", "wine", "perfumery", "seafood", "boutique", "cannabis", "computer", "laundry", "nutrition_supplements", "erotic", "bag", "fabric", "mobile_phone", "frozen_food", "electronics", "farm", "art", "houseware", "beverages", "newsagent", "variety_store", "fishing", "craft", "weapons", "department_store", "water", "baby_goods", "tobacco", "fireplace", "herbalist", "wholesale", "coffee", "kiosk", "candles", "beauty", "hairdresser", "shoes", "toys", "appliance", "vacuum_cleaner", "furnace", "butcher", "doors", "party", "pyrotechnics", "health_food", "bathroom_furnishing", "stationery", "carpet", "doityourself", "bookmaker", "kitchen", "massage", "garden_centre", "tiles", "fashion", "water_sports", "security", "alcohol", "tea", "convenience", "games", "books", "funeral_directors", "hardware", "clothes", "hairdresser_supply", "music", "dry_cleaning", "second_hand", "paint", "copyshop", "sewing", "florist", "gift", "hifi", "pet_grooming", "bed", "spices", "ticket", "hunting", "e-cigarette", "pastry", "chocolate", "medical_supply", "fashion_accessories", "photo", "mall", "general", "cosmetics", "tattoo", "chemist", "watches", "electrical", "trade", "travel_agency", "gas", "curtain", "dairy", "video_games", "pet", "frame", "lighting", "jewelry", "storage_rental", "radiotechnics", "antiques", "lottery", "musical_instrument", "supermarket", "swimming_pool"]}}
service_tags = pd.DataFrame(service_tags)


def verbose_print(text, verbose=True):
    if verbose:
        print(text)


def get_territory(territory_name: str | dict | list[str] | list[dict]):
    territory: Polygon | MultiPolygon

    territory = ox.geocode_to_gdf(territory_name)
    territory = territory.set_crs(4326)
    territory = territory["geometry"].iloc[0]
    return territory


def fetch_buildings(territory: Polygon | MultiPolygon, express_mode=True):
    express_mode: bool
    buildings: GeoDataFrame

    buildings = ox.features_from_polygon(territory, tags={"building": True})
    buildings = buildings.loc[buildings["geometry"].type == "Polygon"]

    if not express_mode:
        buildings_ = ox.features_from_polygon(territory, tags={"building": "yes"})
        buildings_ = buildings_.loc[buildings_["geometry"].type == "Polygon"][
            "geometry"
        ]
        buildings = gpd.GeoSeries(
            pd.concat([buildings, buildings_], ignore_index=True)
        ).drop_duplicates()

    try:
        buildings = (
            buildings[["geometry", "building:levels"]]
            .reset_index(drop=True)
            .rename(columns={"building:levels": "levels"})
        )
    except:
        buildings = buildings["geometry"].reset_index(drop=True)
    return buildings


def fetch_roads(territory: Polygon | MultiPolygon):
    roads: GeoDataFrame

    tags = {
        "highway": [
            "construction",
            "crossing",
            "living_street",
            "motorway",
            "motorway_link" "milestone",
            "motorway_junction",
            "pedestrian",
            "primary",
            "primary_link",
            "proposed",
            "raceway",
            "residential",
            "road",
            "secondary",
            "secondary_link",
            "services",
            "tertiary",
            "tertiary_link",
            "track",
            "trunk",
            "trunk_link",
            "turning_circle",
            "turning_loop",
            "unclassified",
        ],
        "service": ["living_street", "emergency_access"],
    }
    roads = ox.features_from_polygon(territory, tags)
    roads = roads.loc[
        np.logical_or(
            roads["geometry"].type == "LineString",
            roads["geometry"].type == "MultiLineString",
        )
    ]
    roads = roads.reset_index()["geometry"]

    return roads


def fetch_water(territory: Polygon | MultiPolygon):
    water: GeoDataFrame

    try:
        water = ox.features_from_polygon(
            territory, {"natural": ["water","bay"]}
        )  # 'natural':'bay','water':['lake','river','reservoir']
        water = water["geometry"]
        water = water[
            np.logical_or(water.type == "Polygon", water.type == "MultiPolygon")
        ].reset_index(drop=True)
        return water
    except:
        # print('<No water found>')
        return


def fetch_railways(territory: Polygon | MultiPolygon):
    railway: GeoDataFrame

    try:
        railway = ox.features_from_polygon(territory, {"railway": "rail"}).reset_index(
            drop=True
        )
        try:
            railway = railway.query('service not in ["crossover","siding","yard"]')[
                "geometry"
            ]
            return railway
        except:
            return railway["geometry"]
    except:
        # print('<No railways found>')
        return


def fetch_rivers(territory: Polygon | MultiPolygon):
    rivers: GeoDataFrame

    try:
        rivers = ox.features_from_polygon(territory, {"waterway": "river"})
        rivers = rivers["geometry"].reset_index(drop=True)
        return rivers
    except:
        # print('<No rivers found>')
        return


def fill_holes(blocks: GeoDataFrame):
    blocks["geometry"] = blocks["geometry"].boundary
    blocks = blocks.explode(index_parts=False)
    blocks["geometry"] = blocks["geometry"].map(lambda x: Polygon(x))
    blocks = blocks.reset_index(drop=True).to_crs(4326)
    return blocks


def get_overlapping_blocks(blocks: GeoDataFrame) -> list:

    overlaps = blocks["geometry"].sindex.query(blocks["geometry"], predicate="contains")

    overlaps_dict = dict.fromkeys(overlaps[0])
    for x in overlaps_dict:
        overlaps_dict[x] = []

    for x, y in zip(overlaps[0], overlaps[1]):
        if x != y:
            overlaps_dict[x].append(y)

    return list({x for v in overlaps_dict.values() for x in v})


def drop_overlapping_blocks(blocks: GeoDataFrame):

    blocks = blocks.reset_index(drop=True)
    overlapping_block_indeces = get_overlapping_blocks(blocks)
    blocks = blocks.drop(overlapping_block_indeces)
    blocks = blocks.reset_index(drop=True)
    return blocks


def divide_list_into_n_parts(lst: list[DataFrame], n: int):
    res: list[list[DataFrame]]

    if n > len(lst):
        n = len(lst)
    part_length = round(len(lst) / n)

    res = []
    for i in range(1, n):
        res.append(lst[(i - 1) * part_length : i * part_length])

    res.append(lst[(n - 1) * part_length: len(lst)])

    return res


def fetch_services(territory: Polygon | MultiPolygon, service_tags: DataFrame=service_tags, verbose: bool=True):
    res: DataFrame & GeoDataFrame

    res = pd.DataFrame()
    for category in tqdm(service_tags.columns, disable=not verbose):
        tags = dict(service_tags[category].dropna())

        try:
            services_temp = ox.features_from_polygon(territory, tags)
        except InsufficientResponseError:
            continue
        except:
            # for very long queries
            stats = service_tags[category].dropna().map(lambda x: len(x))
            longest_tag = stats.sort_values().index[-1]

            queries = [
                {category: query_part}
                for query_part in divide_list_into_n_parts(
                    service_tags[category].dropna()[longest_tag], 4
                )
            ]
            services_temp = pd.concat(
                [
                    ox.features_from_polygon(territory, query_part)
                    for query_part in queries
                ]
            )

            other_tags = {
                x: service_tags[category][x]
                for x in service_tags[category].dropna().keys()
                if x != longest_tag
            }
            services_temp = pd.concat(
                [services_temp, ox.features_from_polygon(territory, other_tags)]
            )

        try:
            services_temp = pd.concat(
                [
                    services_temp[["name", "geometry"]].reset_index(drop=True),
                    services_temp[tags.keys()]
                    .reset_index(drop=True)
                    .apply(lambda x: list(x.dropna()), axis=1),
                ],
                axis=1,
            )
        except KeyError:
            in_index = list(set(tags.keys()).intersection(services_temp.keys()))
            services_temp = pd.concat(
                [
                    services_temp[["geometry"]].reset_index(drop=True),
                    services_temp[in_index]
                    .reset_index(drop=True)
                    .apply(lambda x: list(x.dropna()), axis=1),
                ],
                axis=1,
            )
        services_temp["category"] = category
        services_temp = services_temp.rename(columns={0: "tags"})

        res = pd.concat([res, services_temp])

    res = gpd.GeoDataFrame(res)
    res["geometry"] = res.to_crs(3857)["geometry"].centroid.to_crs(4326)
    res = res.reset_index(drop=True)

    return res


def get_distance_matrix(geom: GeoDataFrame, verbose: bool=True):  # add geom2 as option
    distance_matrix: DataFrame | Series
    # TODO: create distance matrix not by straight lines but by travel time
    verbose_print("Creating distance matrix...", verbose)

    # coords = geom['geometry'].map(lambda x: [radians(x.x),radians(x.y)]).tolist()
    # distance_matrix = haversine_distances(coords)*6371000

    # geom = geom.to_crs(3857)
    ddf = dask_geopandas.from_geopandas(geom, npartitions=5)
    meta_df = pd.DataFrame(np.nan, index=geom.index, columns=geom.index)
    distance_matrix = ddf["geometry"].apply(
        lambda geom_2: geom.distance(geom_2), meta=meta_df
    )

    return distance_matrix


def cluster_points(distance_matrix: DataFrame | Series,
                   distance_limit: int | float=1000,
                   link: Literal['ward', 'complete', 'average', 'single']="average"):

    clustering: ndarray

    # link can be {‘ward’, ‘complete’, ‘average’, ‘single’}
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        distance_threshold=distance_limit,
        linkage=link,
    ).fit_predict(distance_matrix)
    return clustering


def get_cluster_polygons(services: GeoDataFrame,
                         distance_limit: int | float=1000,
                         link: Literal['ward', 'complete', 'average', 'single']="average",
                         verbose: bool=True):
    cluster_polygons: GeoDataFrame

    distance_matrix = get_distance_matrix(services, verbose)

    services = services.to_crs(4326)
    services["cluster"] = cluster_points(distance_matrix, distance_limit, link)

    services_per_cluster = services.groupby(["cluster"])["geometry"].count()
    hulls = []
    for cluster in services_per_cluster[services_per_cluster > 4].index:
        hulls.append(
            [
                cluster,
                services[services["cluster"] == cluster][
                    "geometry"
                ].unary_union.convex_hull,
            ]
        )

    cluster_polygons = gpd.GeoDataFrame(hulls)
    cluster_polygons.columns = ["cluster", "geometry"]
    cluster_polygons = cluster_polygons.set_geometry("geometry").set_crs(4326)
    cluster_polygons = cluster_polygons[cluster_polygons.type == "Polygon"]

    return cluster_polygons


def get_cluster_blocks(blocks: GeoDataFrame, cluster_polygons: GeoDataFrame):
    blocks["cluster"] = np.nan
    blocks = get_attribute_from_largest_intersection(
        blocks, cluster_polygons, "block_id", "cluster"
    )
    return blocks


def get_diversity(blocks: GeoDataFrame & DataFrame, services: GeoDataFrame, groupping_column: str="cluster"):
    blocks_services = blocks.sjoin(services[["geometry", "service"]])
    diversity = pd.DataFrame(columns=[groupping_column, "diversity"])
    for i, x in (
        blocks_services.groupby(groupping_column)["service"].apply(list).items()
    ):
        diversity.loc[len(diversity)] = [
            i,
            1 - mm.simpson_diversity(pd.Series(x), categorical=True),
        ]
    blocks = blocks.merge(diversity, how="left")
    return blocks


def compress_services(df: DataFrame, method="kmeans", min_samples=2, n_clusters: int=10000):
    df_clustered: DataFrame

    X = (df["geometry"].map(lambda x: [x.x, x.y])).tolist()

    clustering_services = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=42, n_init="auto"
    ).fit(X)

    df["cluster_temp"] = clustering_services.labels_

    df_clustered = (
        gpd.GeoSeries(
            df.groupby(["cluster_temp"])["geometry"].apply(list).apply(MultiPoint)
        )
        .reset_index()
        .set_crs(4326)
    )
    tags_df = df.groupby(["cluster_temp"]).agg({"tags": "sum"}).reset_index()
    df_clustered = df_clustered.merge(tags_df)
    df_clustered["geometry"] = df_clustered.to_crs(3857).centroid.to_crs(4326)
    df_clustered = df_clustered.drop("cluster_temp", axis=1)

    return df_clustered


def get_intersection_area_pivot(df: GeoDataFrame,
                                df_with_attribute: GeoDataFrame,
                                df_id_column: str,
                                attribute_column: str):
    intersection_pivot: DataFrame | None

    df_temp = gpd.overlay(
        df[[df_id_column, "geometry"]],
        df_with_attribute[[attribute_column, "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    df_temp["intersection_area"] = df_temp.to_crs(3857)["geometry"].area
    df_temp = (
        df_temp.groupby([df_id_column, attribute_column])["intersection_area"]
        .sum()
        .reset_index()
    )

    # if 'area' not in df.columns:
    df["area"] = df.to_crs(3857)["geometry"].area
    df_temp = df_temp.merge(df[[df_id_column, "area"]], how="left")
    df_temp["intersection_area"] = df_temp["intersection_area"] / df_temp["area"]
    # df_temp = df_temp.sort_values(by='intersection_area')

    df_groupped = (
        df_temp.groupby([df_id_column, attribute_column])["intersection_area"]
        .sum()
        .reset_index()
    )
    intersection_pivot = pd.pivot_table(
        df_groupped,
        index=df_id_column,
        columns=attribute_column,
        values="intersection_area",
    )
    intersection_pivot = intersection_pivot.replace(0, np.nan)

    return intersection_pivot


def get_n_largest_intersections(intersection_pivot: DataFrame, n: int):
    return intersection_pivot.apply(lambda x: [*x.nlargest(n).dropna()], axis=1)


def get_attribute_from_largest_intersection(
    df: GeoDataFrame & GeoDataFrame,
    df_with_attribute: GeoDataFrame,
    attribute_column: str,
    df_id_column: str="block_id",
    min_intersection:float=0.05,
):
    intersection_pivot: DataFrame | None

    if attribute_column in df.columns:
        df = df.drop(attribute_column, axis=1)

    df_temp = gpd.overlay(
        df[[df_id_column, "geometry"]],
        df_with_attribute[[attribute_column, "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    df_temp["intersection_area"] = df_temp.to_crs(3857)["geometry"].area
    df_temp = (
        df_temp.groupby([df_id_column, attribute_column])["intersection_area"]
        .sum()
        .reset_index()
        .sort_values([df_id_column, "intersection_area"], ascending=False)
        .drop_duplicates(subset=df_id_column, keep="first")
    )

    df["area"] = df.to_crs(3857)["geometry"].area
    df_temp = df_temp.merge(df[[df_id_column, "area"]], on=df_id_column, how="left")
    df_temp["intersection_area"] = df_temp["intersection_area"] / df_temp["area"]

    df_temp = df_temp.sort_values(by="intersection_area")
    df_temp = df_temp.drop_duplicates(subset=df_id_column, keep="last")

    df = df.merge(
        df_temp[[df_id_column, attribute_column, "intersection_area"]], how="left"
    )

    df.loc[df["intersection_area"] < min_intersection, attribute_column] = np.nan
    df = df.drop("intersection_area", axis=1)

    return df

def reindex_blocks(blocks: GeoDataFrame & DataFrame):
    blocks = blocks.drop('block_id',axis=1).reset_index().rename(columns={'index':'block_id'})
    return blocks


def filter_small_blocks(blocks: GeoDataFrame & DataFrame, crs: pyproj.CRS=3857, min_width: int=40):

    blocks_flt = (
        blocks.to_crs(crs)
        .buffer(-min_width / 2, cap_style=3)
        .buffer(min_width / 2, cap_style=3)
        .explode(index_parts=False)
        .to_crs(4326)
    )
    blocks_flt = blocks_flt[[x != Polygon() for x in blocks_flt]]
    blocks = blocks.loc[blocks_flt.index]

    return blocks


class InfrastructureCityModel(CityModel):
    # TODO: fix cluster assignment. Merge blocks with cluster polys than take the most area
    # TODO: write setters for class attributes
    # TODO: add saver
    # TODO: unclutter CityModel.blocks attribute by creating a new attribute that keeps cluster information such as diversity, services_total, centrality, cluster_area, etc.

    def __init__(
        self,
        territory: Polygon | MultiPolygon | str,
        roads: gpd.GeoDataFrame | gpd.GeoSeries | np.nan=np.nan,
        water: gpd.GeoDataFrame | gpd.GeoSeries | np.nan=np.nan,
        rivers: gpd.GeoDataFrame | gpd.GeoSeries | np.nan=np.nan,
        railways: gpd.GeoDataFrame | gpd.GeoSeries | np.nan=np.nan,
        buildings: gpd.GeoDataFrame | gpd.GeoSeries | np.nan=np.nan,
        nodev: gpd.GeoDataFrame | gpd.GeoSeries | np.nan=np.nan,
        skip_buildings: bool=True,
        filter_small: bool=True,
        territory_name: str="Unnamed Territory",
        verbose: bool=True,
    ):

        self.verbose = verbose
        self.territory_name = territory_name
        if territory_name != "Unnamed Territory":
            self.territory_name = territory_name
        if type(territory) == str:
            self.territory_name = territory
            self.territory = get_territory(territory)
        elif type(territory) in [Polygon, MultiPolygon]:
            self.territory = territory

        self.local_crs = CRS(3857)

        try:
            verbose_print("Setting local CRS...", self.verbose)
            self.set_local_crs()
        except:
            verbose_print("Setting local CRS failed.", self.verbose)

        if type(roads) not in [gpd.GeoDataFrame, gpd.GeoSeries]:
            verbose_print("Downloading roads...", self.verbose)
            roads = fetch_roads(self.territory)
        self.roads = roads

        if type(water) not in [gpd.GeoDataFrame, gpd.GeoSeries]:
            verbose_print("Downloading water...", self.verbose)
            water = fetch_water(self.territory)
        self.water = water

        if type(rivers) not in [gpd.GeoDataFrame, gpd.GeoSeries]:
            rivers = fetch_rivers(self.territory)
        self.rivers = rivers

        if type(railways) not in [gpd.GeoDataFrame, gpd.GeoSeries]:
            verbose_print("Downloading railways...", self.verbose)
            railways = fetch_railways(self.territory)
        self.railways = railways

        if (
            type(buildings) not in [gpd.GeoDataFrame, gpd.GeoSeries]
            and skip_buildings == False
        ):
            verbose_print("Downloading buildings...", self.verbose)
            buildings = fetch_buildings(self.territory)
        self.buildings = buildings

        self.blocks = np.nan
        self.services = np.nan
        self.cluster_polygons = np.nan
        self.cluster_info = np.nan

        verbose_print("CityModel initialized.", self.verbose)

    def set_local_crs(self) -> None:
        coords = [list(set(x)) for x in self.territory.envelope.boundary.coords.xy]

        area_of_interest = AreaOfInterest(
            west_lon_degree=coords[0][0],
            east_lon_degree=coords[0][1],
            north_lat_degree=coords[1][0],
            south_lat_degree=coords[1][1],
        )

        utm_crs_list = query_crs_info(
            pj_types=pyproj.enums.PJType.PROJECTED_CRS,
            area_of_interest=area_of_interest,
        )
        utm_crs = CRS.from_epsg(utm_crs_list[0].code)

        self.local_crs = utm_crs

    def set_services(self, service_tags=service_tags, services=np.nan):
        service_tags: DataFrame
        service_tags: np.nan | float

        if type(services) == float:
            verbose_print("Downloading services...", self.verbose)
            self.services = fetch_services(
                self.territory, service_tags, verbose=self.verbose
            )
        else:
            self.services = services
        verbose_print("Associating services with blocks...", self.verbose)
        self.associate_services_with_blocks()
        verbose_print("Services set.", self.verbose)

    def associate_services_with_blocks(self) -> None:
        blocks_columns = ["block_id", "geometry"]
        if "cluster" in self.blocks.columns:
            blocks_columns.append("cluster")

        if "block_id" in self.services.columns:
            self.services = self.services.drop("block_id", axis=1)
        if "cluster" in self.services.columns:
            self.services = self.services.drop("cluster", axis=1)

        self.services = (
            self.services.to_crs(3857)
            .sjoin_nearest(
                self.blocks.to_crs(3857)[blocks_columns], how="left", max_distance=200
            )
            .to_crs(4326)
        )
        self.services = self.services.dropna(subset="block_id")
        self.services = self.services.drop("index_right", axis=1)
        self.services["block_id"] = self.services["block_id"].astype(int)
        self.services = self.services.reset_index(drop=True)

    def generate_blocks(
        self, filter_small=True, save_file=False, filename="blocks.geojson"
    ):
        filter_small: bool
        save_file: bool
        filename: str

        verbose_print("Generating blocks...", self.verbose)
        limit = gpd.GeoDataFrame(geometry=[self.territory.boundary]).set_crs(4326)
        lines = (
            gpd.GeoDataFrame(
                geometry=pd.concat(
                    [self.roads, self.rivers, self.railways, self.water.boundary]
                )
            )
            .set_crs(4326)
            .reset_index(drop=True)
        )
        lines = lines.explode(index_parts=True).reset_index(drop=True)

        verbose_print("Setting up enclosures...", self.verbose)
        blocks = mm.enclosures(lines, limit=limit, enclosure_id="index").drop(
            "index", axis=1
        )

        verbose_print("Cutting water...", self.verbose)
        blocks = gpd.overlay(
            blocks,
            gpd.GeoDataFrame(self.water.reset_index()),
            how="difference",
            keep_geom_type=False,
        )

        verbose_print("Dropping overlapping blocks...", self.verbose)
        blocks = blocks.explode(index_parts=False)
        blocks = blocks.drop_duplicates(subset="geometry")

        verbose_print("Filtering small blocks...", self.verbose)
        blocks = blocks.reset_index().rename(columns={"index": "block_id"})
        if filter_small:
            blocks = filter_small_blocks(blocks, crs=self.local_crs, min_width=40)

        verbose_print("Calculating blocks area...", self.verbose)
        blocks["area"] = blocks.to_crs(self.local_crs).area
        blocks = blocks[blocks["area"] > 1]

        blocks = (
            blocks.drop("block_id", axis=1)
            .reset_index()
            .rename(columns={"index": "block_id"})
        )

        blocks = blocks.drop_duplicates(subset="geometry")
        blocks = reindex_blocks(blocks)

        self.blocks = blocks
        verbose_print("Blocks generated.", self.verbose)

    def filter_blocks_without_buildings(self) -> None:
        if type(self.buildings) == float:
            verbose_print("Fetching buildings...", self.verbose)
            self.buildings = fetch_buildings(self.territory)

        verbose_print("Filtering blocks without buildings...", self.verbose)
        self.blocks = self.blocks.loc[
            list(
                set(
                    self.blocks.sindex.query(
                        self.buildings["geometry"], predicate="covered_by"
                    )[1]
                )
            )
        ]
        self.blocks = self.blocks.reset_index(drop=True)

    def cut_nodev(self, nodev) -> None:
        nodev: GeoDataFrame

        verbose_print(f'Adding nodev zones to blocks...',self.verbose)

        if type(self.blocks) == float:
            raise Exception(
                "Blocks are not generated. Generate blocks first with generate_blocks() method."
            )

        blocks_new = gpd.overlay(
            self.blocks,
            nodev,
            how="difference",
            keep_geom_type=False,
        )

        blocks_new = blocks_new.explode(index_parts=False)
        blocks_new = (
            blocks_new.drop("block_id", axis=1)
            .reset_index(drop=True)
            .reset_index()
            .rename(columns={"index": "block_id"})
        )

        blocks_new = filter_small_blocks(blocks_new, self.local_crs)

        blocks_new = pd.concat([blocks_new, nodev[["geometry", "CODE_ZONE_"]]])
        blocks_new = blocks_new.explode(index_parts=False)
        # blocks_new['block_id'] = blocks_new['block_id'].fillna(-1)
        blocks_new = blocks_new.drop_duplicates(subset="geometry")
        blocks_new["nodev"] = blocks_new["CODE_ZONE_"].notna()
        blocks_new = (
            blocks_new.drop("block_id", axis=1)
            .reset_index()
            .rename(columns={"index": "block_id"})
        )

        self.blocks = blocks_new
        verbose_print(f'Nodev zones added.',self.verbose)

    def add_pzz(self,pzz: Any,columns: dict ={'CODE_ZONE_':'zone','CODE_VID_Z':'zone_class'},min_intersection: float=0.05):
        pzz: Any
        columns: dict
        min_intersection: float
        verbose_print(f'Adding attributes {list(columns.keys())} from pzz...',self.verbose)
        blocks_pzz = self.blocks.copy()

        for pzz_attribute,attribute_name in columns.items():
            blocks_pzz = get_attribute_from_largest_intersection(blocks_pzz,pzz,pzz_attribute,min_intersection=min_intersection)
            blocks_pzz = blocks_pzz.rename(columns={pzz_attribute:attribute_name})
            blocks_pzz[f'has_{attribute_name}'] = blocks_pzz[attribute_name].notna()

        self.blocks = blocks_pzz
        verbose_print(f'pzz attributes added.',self.verbose)


def get_bounding_polygon(gdf,crs=4326):
    all_polygons_gdf = gdf.make_valid().unary_union
    all_polygons_gdf = fill_holes(gpd.GeoDataFrame(geometry=[all_polygons_gdf],crs=crs))
    all_polygons_gdf['area'] = all_polygons_gdf.to_crs(3857).area
    bounding_polygon = all_polygons_gdf.sort_values('area',ascending=False).iloc[0]['geometry']
    return bounding_polygon
#
#
# pzz = gpd.read_file('pzz_2019.geojson').to_crs(4326)
# nodev = gpd.read_file('no_development_pzz.geojson')
#
# ter = get_bounding_polygon(pzz)
#
# spb = CityModel(territory=ter, territory_name='Санкт-Петербург')
# spb.generate_blocks()
# spb.cut_nodev(nodev)
# spb.add_pzz(pzz)
#
# # pip install folium matplotlib mapclassify
#
# spb.blocks.explore()
