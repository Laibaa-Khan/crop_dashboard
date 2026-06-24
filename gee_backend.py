import ee

ee.Initialize(
    project="bubbly-sentinel-486808-v7"
)

# ==========================
# PROVINCE REGION
# ==========================

def get_region(province):

    mapping = {
        "Khyber Pakhtunkhwa":
        "North-West Frontier"
    }

    province = mapping.get(
        province,
        province
    )

    region = (
        ee.FeatureCollection(
            "FAO/GAUL/2015/level1"
        )
        .filter(
            ee.Filter.eq(
                "ADM0_NAME",
                "Pakistan"
            )
        )
        .filter(
            ee.Filter.eq(
                "ADM1_NAME",
                province
            )
        )
        .geometry()
    )

    return region


# ==========================
# CROP SEASONS
# ==========================

def get_season(crop):

    seasons = {

        "Wheat":
        ("2023-11-01", "2024-03-31"),

        "Rice":
        ("2023-07-01", "2023-10-31"),

        "Cotton":
        ("2023-05-01", "2023-10-31"),

        "Maize":
        ("2023-06-01", "2023-09-30"),

        "Sugarcane":
        ("2023-04-01", "2023-12-31")
    }

    return seasons.get(
        crop,
        ("2023-01-01", "2023-12-31")
    )


# ==========================
# CSV TO EARTH ENGINE
# ==========================

def csv_to_ee(df):

    features = []

    for _, row in df.iterrows():

        feature = ee.Feature(

            ee.Geometry.Point([
                float(row["longitude"]),
                float(row["latitude"])
            ]),

            {
                "Description":
                str(row["crop"]).strip(),

                "Province":
                str(row["province"]).strip()
            }
        )

        features.append(feature)

    return ee.FeatureCollection(
        features
    )


# ==========================
# TRAINING DATA
# ==========================

def get_training(
    table,
    region,
    crop,
    province=None
):

    if province:

        data = table.filter(
            ee.Filter.eq(
                "Province",
                province
            )
        )

    else:

        data = table.filterBounds(
            region
        )

    crop_points = data.filter(
        ee.Filter.eq(
            "Description",
            crop
        )
    )

    count = crop_points.size().getInfo()

    if count == 0:

        raise Exception(
            f"No {crop} data found in {province}"
        )

    crop_points = crop_points.map(
        lambda f:
        f.set("class", 1)
    )

    crop_points = crop_points.limit(300)

    background = (
        ee.FeatureCollection.randomPoints(
            region=region,
            points=300,
            seed=42
        )
        .map(lambda f: f.set("class",0))
    )

    return crop_points.merge(
        background
    )


# ==========================
# SATELLITE IMAGE
# ==========================

def build_image(
    region,
    start,
    end
):

    bands = [
        "B2",
        "B3",
        "B4",
        "B8",
        "B11"
    ]

    collection = (

        ee.ImageCollection(
            "COPERNICUS/S2_SR_HARMONIZED"
        )

        .filterBounds(region)

        .filterDate(
            start,
            end
        )

        .filter(
            ee.Filter.lt(
                "CLOUDY_PIXEL_PERCENTAGE",
                90
            )
        )
    )

    size = collection.size()

    image = ee.Image(

        ee.Algorithms.If(

            size.gt(0),

            collection.median(),

            ee.Image.constant(
                [0, 0, 0, 0, 0]
            ).rename(
                bands
            )
        )
    )

    return image.select(
        bands
    )


# ==========================
# RANDOM FOREST
# ==========================

def run_rf(
    region,
    train,
    start,
    end
):

    bands = [
        "B2",
        "B3",
        "B4",
        "B8",
        "B11"
    ]

    img = build_image(
        region,
        start,
        end
    )

    samples = (

        img.sampleRegions(
            collection=train,
            properties=["class"],
            scale=10
        )

        .randomColumn(
            "random"
        )
    )

    train_set = samples.filter(
        ee.Filter.lt(
            "random",
            0.7
        )
    )

    test_set = samples.filter(
        ee.Filter.gte(
            "random",
            0.7
        )
    )

    classifier = (

        ee.Classifier

        .smileRandomForest(
            50
        )

        .train(
            train_set,
            "class",
            bands
        )
    )

    accuracy = (

        test_set

        .classify(
            classifier
        )

        .errorMatrix(
            "class",
            "classification"
        )

        .accuracy()
    )

    classified = (

        img

        .classify(
            classifier
        )

        .clip(
            region
        )
    )

    return (
        classified,
        accuracy.getInfo()
    )


# ==========================
# SVM
# ==========================

def run_svm(
    region,
    train,
    start,
    end
):

    bands = [
        "B2",
        "B3",
        "B4",
        "B8",
        "B11"
    ]

    img = build_image(
        region,
        start,
        end
    )

    samples = (

        img.sampleRegions(
            collection=train,
            properties=["class"],
            scale=10
        )

        .randomColumn(
            "random"
        )
    )

    train_set = samples.filter(
        ee.Filter.lt(
            "random",
            0.7
        )
    )

    test_set = samples.filter(
        ee.Filter.gte(
            "random",
            0.7
        )
    )

    classifier = (

        ee.Classifier

        .libsvm(

            kernelType="RBF",

            gamma=0.5,

            cost=10
        )

        .train(
            train_set,
            "class",
            bands
        )
    )

    accuracy = (

        test_set

        .classify(
            classifier
        )

        .errorMatrix(
            "class",
            "classification"
        )

        .accuracy()
    )

    classified = (

        img

        .classify(
            classifier
        )

        .clip(
            region
        )
    )

    return (
        classified,
        accuracy.getInfo()
    )
    