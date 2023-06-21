# SPDX-License-Identifier: Apache-2.0
from mlmonitor.src import API_KEY, WOS_URL


def custom_metrics_provider_gcr(parms={"url": WOS_URL, "apikey": API_KEY}):
    url = parms.get("url")
    apikey = parms.get("apikey")
    from custmonitor.metricsprovider.helpers import publish

    # Add your code to compute the custom metrics here.
    # Based on use case , you can inlude get_metrics from custmonitor.metrics.<use case>
    # - credit_risk   from custmonitor.metrics.credit_risk import get_metrics
    # - churn         from custmonitor.metrics.customer_churn import get_metrics
    # or other use case
    from custmonitor.metrics.use_case_gcr import get_metrics

    def publish_to_monitor(input_data):
        response_payload = publish(
            input_data=input_data, url=url, apikey=apikey, get_metrics_fn=get_metrics
        )
        return response_payload

    return publish_to_monitor


def custom_metrics_provider_churn(parms={"url": WOS_URL, "apikey": API_KEY}):
    url = parms.get("url")
    apikey = parms.get("apikey")
    from custmonitor.metricsprovider.helpers import publish

    # Add your code to compute the custom metrics here.
    # Based on use case , you can include get_metrics from custmonitor.metrics.<use case>
    # - credit_risk   from custmonitor.metrics.credit_risk import get_metrics
    # - churn         from custmonitor.metrics.customer_churn import get_metrics
    # or other use case
    from custmonitor.metrics.use_case_churn import get_metrics

    def publish_to_monitor(input_data):
        response_payload = publish(
            input_data=input_data, url=url, apikey=apikey, get_metrics_fn=get_metrics
        )
        return response_payload

    return publish_to_monitor
