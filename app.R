library(shiny)
library(readr)
library(tsibble)
library(tidyr)
library(dplyr)
library(feasts)
library(zoo)
library(fable)
library(fabletools)
library(ggplot2)
library(bslib)
# -------------------------------------------------------------------
# 1. Load + prepare data  (FAST + clean)
# -------------------------------------------------------------------

raw_wines <- read_csv("AustralianWines.csv", show_col_types = FALSE)

raw_wines <- raw_wines |>
  mutate(
    Rose  = as.numeric(Rose),
    Month = yearmonth(as.yearmon(Month, "%b-%y"))
  )

# Long → tsibble (do this ONCE)
wines_ts <- raw_wines |>
  pivot_longer(cols = -Month,
               names_to  = "Varietal",
               values_to = "Sales") |>
  as_tsibble(index = Month, key = Varietal)

var_choices <- sort(unique(wines_ts$Varietal))
min_date <- as.Date(min(wines_ts$Month))
max_date <- as.Date(max(wines_ts$Month))


# -------------------------------------------------------------------
# 2. UI  (same layout as your version)
# -------------------------------------------------------------------



ui <- page_navbar(
  
  title = "Australian Wine Forecasting",
  theme = bs_theme(
    version = 5,
    bootswatch = "flatly",        # modern + lightweight theme
    primary = "#5A3E85",
    secondary = "#E0E0E0",
    base_font = font_google("Inter"),
    heading_font = font_google("Poppins")
  ),
  
  nav_panel(
    "Data",
    
    layout_sidebar(
      sidebar = sidebar(
        width = "300px",
        card(
          card_header("Filters"),
          selectInput("varietal", "Varietal:", choices = var_choices),
          dateRangeInput("date_range", "Date Range:",
                         start = min_date, end = max_date,
                         min = min_date, max = max_date),
          numericInput("h", "Forecast Horizon (Months):",
                       value = 12, min = 3, max = 60),
          helpText("Training = all but last h months. Validation = last h.")
        )
      ),
      
      card(
        full_screen = TRUE,
        card_header("Historical Sales"),
        plotOutput("sales_plot", height = "450px")
      )
    )
  ),
  
  nav_panel(
    "Seasonality",
    card(
      full_screen = TRUE,
      card_header("Seasonal Pattern"),
      plotOutput("season_plot", height = "450px")
    )
  ),
  
  nav_panel(
    "Forecasts",
    card(
      full_screen = TRUE,
      card_header("Forecast with Prediction Intervals"),
      plotOutput("fc_plot", height = "450px")
    )
  ),
  
  nav_panel(
    "Accuracy",
    layout_columns(
      card(
        width = 6,
        card_header("Accuracy Table"),
        tableOutput("acc_table")
      ),
      card(
        width = 6,
        card_header("Model Specifications"),
        verbatimTextOutput("model_specs")
      )
    )
  )
)


# -------------------------------------------------------------------
# 3. SERVER  (Optimized + Minor Fixes)
# -------------------------------------------------------------------

server <- function(input, output, session) {

  # FAST filtered data
  filtered_data <- reactive({
    req(input$date_range)
    wines_ts |>
      filter(
        Varietal == input$varietal,
        Month >= yearmonth(input$date_range[1]),
        Month <= yearmonth(input$date_range[2])
      )
  })

  # Train / validation split
  train_valid <- reactive({
    dat <- filtered_data()
    h <- input$h

    req(nrow(dat) > h + 5)

    n <- nrow(dat)
    train <- dat |> slice_head(n = n - h)
    valid <- dat |> slice_tail(n = h)

    list(train = train, valid = valid)
  })

  # MODELS (slightly adjusted)
  models <- reactive({
    tv <- train_valid()
    train <- tv$train

    # ARIMA allowed — but safe tryCatch to avoid failures
    try_models <- try({
      train |>
        model(
          TSLM  = TSLM(Sales ~ trend() + season()),
          ETS   = ETS(Sales),
          ARIMA = ARIMA(Sales)
        )
    }, silent = TRUE)

    # If ARIMA failed → fit only ETS + TSLM
    if (inherits(try_models, "try-error")) {
      train |>
        model(
          TSLM  = TSLM(Sales ~ trend() + season()),
          ETS   = ETS(Sales)
        )
    } else {
      try_models
    }
  })

  # Forecasts for h months
  forecasts <- reactive({
    models() |>
      forecast(h = input$h)
  })

  # -------------------------------------------------------------------
  # PLOTS
  # -------------------------------------------------------------------

  # History
  output$sales_plot <- renderPlot({
    filtered_data() |>
      autoplot(Sales) +
      labs(
        title = paste("Historical sales:", input$varietal),
        x = "Month", y = "Sales"
      )
  })

  # Seasonality
  output$season_plot <- renderPlot({
    filtered_data() |>
      gg_season(Sales) +
      labs(
        title = paste("Seasonal pattern:", input$varietal),
        x = "Month", y = "Sales"
      )
  })

  # Forecast plot
  output$fc_plot <- renderPlot({
    tv <- train_valid()
    train <- tv$train
    valid <- tv$valid
    fc <- forecasts()

    autoplot(fc, train) +
      autolayer(valid, Sales, colour = "black", linetype = "dashed") +
      labs(
        title = paste("Forecasts for:", input$varietal),
        caption = "Dashed = validation data",
        x = "Month", y = "Sales"
      )
  })


  # -------------------------------------------------------------------
  # ACCURACY TABLES
  # -------------------------------------------------------------------

  output$acc_table <- renderTable({
    tv <- train_valid()
    train <- tv$train
    valid <- tv$valid
    mdl <- models()

    # TRAIN accuracy — safe for all models
    acc_train <- mdl |> accuracy()

    # VALIDATION accuracy — reforecast properly
    fc_valid <- mdl |> forecast(h = nrow(valid))
    acc_valid <- try(accuracy(fc_valid, valid), silent = TRUE)

    # If ARIMA fails → remove it
    if (inherits(acc_valid, "try-error")) {
      mdl2 <- mdl %>% select(-ARIMA)
      fc_valid <- mdl2 |> forecast(h = nrow(valid))
      acc_valid <- accuracy(fc_valid, valid)
    }

    acc_train$Set <- "Train"
    acc_valid$Set <- "Validation"

    bind_rows(
      acc_train |> select(Set, .model, RMSE, MAE, MAPE),
      acc_valid |> select(Set, .model, RMSE, MAE, MAPE)
    )
  })

  # Model specs
  output$model_specs <- renderPrint({
    report(models())
  })
}

# -------------------------------------------------------------------
# Run App
# -------------------------------------------------------------------
shinyApp(ui, server)
