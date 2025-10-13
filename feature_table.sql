-- Drop and recreate with all features
DROP TABLE IF EXISTS public.feature_house;

CREATE TABLE public.feature_house (
    quarter DATE PRIMARY KEY,
    year INTEGER,
    quarter_num TEXT,  -- Q1, Q2, Q3, Q4
    
    -- Seasonal features (numeric)
    quarter_number INTEGER,  -- 1, 2, 3, 4
    quarter_sin DOUBLE PRECISION,  -- sin encoding for cyclical patterns
    quarter_cos DOUBLE PRECISION,  -- cos encoding for cyclical patterns
    
    -- Original values (DO NOT USE AS FEATURES - potential leakage)
    house_sales DOUBLE PRECISION,
    hpi DOUBLE PRECISION,
    house_stock DOUBLE PRECISION,
    residential_investment DOUBLE PRECISION,
    ocr DOUBLE PRECISION,
    cpi DOUBLE PRECISION,
    gdp DOUBLE PRECISION,
    
    -- hpi_growth and related features
    hpi_growth DOUBLE PRECISION,
    hpi_growth_lag1 DOUBLE PRECISION,
    hpi_growth_lag2 DOUBLE PRECISION,
    hpi_growth_lag3 DOUBLE PRECISION,
    hpi_growth_lag4 DOUBLE PRECISION,
    hpi_growth_lag16 DOUBLE PRECISION,
    hpi_growth_rolling_mean_1y DOUBLE PRECISION,
    hpi_growth_rolling_mean_4y DOUBLE PRECISION,
    hpi_growth_rolling_mean_10y DOUBLE PRECISION,
    hpi_growth_diff_lag1_minus_lag2 DOUBLE PRECISION,
    hpi_growth_diff_lag1_minus_lag4 DOUBLE PRECISION,
    hpi_growth_ratio_lag1_over_lag2 DOUBLE PRECISION,
    hpi_growth_ratio_lag1_over_lag4 DOUBLE PRECISION,
    
    -- house_sales features
    house_sales_lag1 DOUBLE PRECISION,
    house_sales_lag2 DOUBLE PRECISION,
    house_sales_lag3 DOUBLE PRECISION,
    house_sales_lag4 DOUBLE PRECISION,
    house_sales_lag16 DOUBLE PRECISION,
    house_sales_rolling_mean_1y DOUBLE PRECISION,
    house_sales_rolling_mean_4y DOUBLE PRECISION,
    house_sales_rolling_mean_10y DOUBLE PRECISION,
    house_sales_diff_lag1_minus_lag2 DOUBLE PRECISION,
    house_sales_diff_lag1_minus_lag4 DOUBLE PRECISION,
    house_sales_ratio_lag1_over_lag2 DOUBLE PRECISION,
    house_sales_ratio_lag1_over_lag4 DOUBLE PRECISION,
    
    -- lag1 features for base variables
    hpi_lag1 DOUBLE PRECISION,
    house_stock_lag1 DOUBLE PRECISION,
    residential_investment_lag1 DOUBLE PRECISION,
    ocr_lag1 DOUBLE PRECISION,
    cpi_lag1 DOUBLE PRECISION,
    gdp_lag1 DOUBLE PRECISION,
    
    -- rolling means for other variables
    hpi_rolling_mean_1y DOUBLE PRECISION,
    hpi_rolling_mean_4y DOUBLE PRECISION,
    hpi_rolling_mean_10y DOUBLE PRECISION,
    house_stock_rolling_mean_1y DOUBLE PRECISION,
    house_stock_rolling_mean_4y DOUBLE PRECISION,
    house_stock_rolling_mean_10y DOUBLE PRECISION,
    residential_investment_rolling_mean_1y DOUBLE PRECISION,
    residential_investment_rolling_mean_4y DOUBLE PRECISION,
    residential_investment_rolling_mean_10y DOUBLE PRECISION,
    ocr_rolling_mean_1y DOUBLE PRECISION,
    ocr_rolling_mean_4y DOUBLE PRECISION,
    ocr_rolling_mean_10y DOUBLE PRECISION,
    cpi_rolling_mean_1y DOUBLE PRECISION,
    cpi_rolling_mean_4y DOUBLE PRECISION,
    cpi_rolling_mean_10y DOUBLE PRECISION,
    gdp_rolling_mean_1y DOUBLE PRECISION,
    gdp_rolling_mean_4y DOUBLE PRECISION,
    gdp_rolling_mean_10y DOUBLE PRECISION,

    -- Policy flags
    covid_lockdown_2020q2_q3 INTEGER NOT NULL DEFAULT 0,
    reopening_supply_2021q2_2022q4 INTEGER NOT NULL DEFAULT 0,

    -- Scaled features (z-score normalized from lag1 values - CLEARLY LEAKAGE-FREE)
    hpi_lag1_scaled DOUBLE PRECISION,
    house_sales_lag1_scaled DOUBLE PRECISION,
    hpi_growth_lag1_scaled DOUBLE PRECISION,
    gdp_lag1_scaled DOUBLE PRECISION,
    house_stock_lag1_scaled DOUBLE PRECISION,
    residential_investment_lag1_scaled DOUBLE PRECISION,
    ocr_lag1_scaled DOUBLE PRECISION,
    cpi_lag1_scaled DOUBLE PRECISION,

    CONSTRAINT chk_feature_house_binary_flags CHECK (
      covid_lockdown_2020q2_q3 IN (0,1)
      AND reopening_supply_2021q2_2022q4 IN (0,1)
    )
);

-- Index and comments
CREATE INDEX idx_feature_house_year ON public.feature_house(year);
COMMENT ON TABLE public.feature_house IS 'Feature-engineered house data - all features are leakage-free';
COMMENT ON COLUMN public.feature_house.quarter_number IS 'Quarter as integer (1-4) for seasonal patterns';
COMMENT ON COLUMN public.feature_house.quarter_sin IS 'Sin-encoded quarter for cyclical seasonal patterns';
COMMENT ON COLUMN public.feature_house.quarter_cos IS 'Cos-encoded quarter for cyclical seasonal patterns';
COMMENT ON COLUMN public.feature_house.covid_lockdown_2020q2_q3 IS '1 for 2020-04-01..2020-09-30, else 0';
COMMENT ON COLUMN public.feature_house.reopening_supply_2021q2_2022q4 IS '1 for 2021-04-01..2022-12-31, else 0';
COMMENT ON COLUMN public.feature_house.hpi_lag1_scaled IS 'Z-score normalized from hpi_lag1 (leakage-free)';
COMMENT ON COLUMN public.feature_house.house_sales_lag1_scaled IS 'Z-score normalized from house_sales_lag1 (leakage-free)';
COMMENT ON COLUMN public.feature_house.gdp_lag1_scaled IS 'Z-score normalized from gdp_lag1 (leakage-free)';
COMMENT ON COLUMN public.feature_house.house_stock_lag1_scaled IS 'Z-score normalized from house_stock_lag1 (leakage-free)';
COMMENT ON COLUMN public.feature_house.residential_investment_lag1_scaled IS 'Z-score normalized from residential_investment_lag1 (leakage-free)';
COMMENT ON COLUMN public.feature_house.ocr_lag1_scaled IS 'Z-score normalized from ocr_lag1 (leakage-free)';
COMMENT ON COLUMN public.feature_house.cpi_lag1_scaled IS 'Z-score normalized from cpi_lag1 (leakage-free)';
COMMENT ON COLUMN public.feature_house.hpi_growth_lag1_scaled IS 'Z-score normalized from hpi_growth_lag1 (leakage-free)';
