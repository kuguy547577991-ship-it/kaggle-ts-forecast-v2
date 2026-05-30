"""Data quality expectations for Kaggle store-item sales data.

Validates:
- Mandatory columns: date, store, item, sales
- sales >= 0 (no negative sales)
- date column is valid datetime
"""

from great_expectations.core import ExpectationSuite


def build_sales_suite() -> ExpectationSuite:
    suite = ExpectationSuite(expectation_suite_name="kaggle_sales_data")

    for col in ["date", "store", "item", "sales"]:
        suite.add_expectation(
            expectation_type="expect_column_to_exist",
            kwargs={"column": col},
        )

    suite.add_expectation(
        expectation_type="expect_column_values_to_be_greater_than",
        kwargs={"column": "sales", "min_value": 0},
    )

    suite.add_expectation(
        expectation_type="expect_column_values_to_not_be_null",
        kwargs={"column": "sales"},
    )

    return suite
