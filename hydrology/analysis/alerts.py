"""
Real-time alert system for hydrology monitoring.

Provides threshold-based monitoring for discharge, stage, and other
parameters with configurable notification options.

Features:
- Define alert thresholds (flood stage, low flow, rate of change)
- Check current conditions against thresholds
- Send notifications via email, webhook, or logging
- Track alert history

Example:
    >>> from hydrology.analysis.alerts import AlertMonitor, AlertThreshold
    >>> monitor = AlertMonitor()
    >>> monitor.add_threshold(AlertThreshold(
    ...     site_id='12422500',
    ...     parameter='discharge',
    ...     condition='above',
    ...     value=10000,
    ...     severity='warning',
    ...     message='High flow warning at Spokane River'
    ... ))
    >>> alerts = monitor.check_site('12422500')
"""

import json
import smtplib
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
import pandas as pd
import requests

from ..core.logging_setup import get_logger
from ..core.paths import ensure_dir, OUTPUT_DIR
from ..data.usgs import fetch_daily_values, fetch_instantaneous_values

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertCondition(Enum):
    """Conditions for triggering alerts."""
    ABOVE = "above"
    BELOW = "below"
    EQUALS = "equals"
    RATE_INCREASE = "rate_increase"  # Rate of change > threshold
    RATE_DECREASE = "rate_decrease"  # Rate of change < -threshold


@dataclass
class AlertThreshold:
    """
    Definition of an alert threshold.

    Attributes:
        site_id: USGS site ID to monitor
        parameter: Parameter to monitor ('discharge', 'stage', 'temp')
        condition: Condition type (above, below, rate_increase, etc.)
        value: Threshold value
        severity: Alert severity level
        message: Custom alert message
        enabled: Whether the alert is active
        cooldown_minutes: Minimum time between repeated alerts
        param_code: USGS parameter code (defaults based on parameter name)
    """
    site_id: str
    parameter: str
    condition: str
    value: float
    severity: str = "warning"
    message: str = ""
    enabled: bool = True
    cooldown_minutes: int = 60
    param_code: Optional[str] = None

    def __post_init__(self):
        """Set default param_code based on parameter name."""
        if self.param_code is None:
            param_map = {
                'discharge': '00060',
                'stage': '00065',
                'gage_height': '00065',
                'temperature': '00010',
                'temp': '00010'
            }
            self.param_code = param_map.get(self.parameter.lower(), '00060')


@dataclass
class Alert:
    """
    A triggered alert instance.

    Attributes:
        threshold: The threshold that was exceeded
        current_value: Current value that triggered the alert
        previous_value: Previous value (for rate-based alerts)
        timestamp: When the alert was triggered
        acknowledged: Whether the alert has been acknowledged
        notified: Whether notifications were sent
    """
    threshold: AlertThreshold
    current_value: float
    timestamp: datetime
    previous_value: Optional[float] = None
    acknowledged: bool = False
    notified: bool = False
    alert_id: str = field(default_factory=lambda: datetime.now().strftime('%Y%m%d%H%M%S%f'))

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        return {
            'alert_id': self.alert_id,
            'site_id': self.threshold.site_id,
            'parameter': self.threshold.parameter,
            'condition': self.threshold.condition,
            'threshold_value': self.threshold.value,
            'current_value': self.current_value,
            'previous_value': self.previous_value,
            'severity': self.threshold.severity,
            'message': self.threshold.message,
            'timestamp': self.timestamp.isoformat(),
            'acknowledged': self.acknowledged,
            'notified': self.notified
        }


class AlertNotifier:
    """
    Handles sending notifications for alerts.

    Supports multiple notification methods:
    - Logging (always enabled)
    - Email (requires SMTP configuration)
    - Webhook (POST to URL)
    - Custom callback functions
    """

    def __init__(self):
        self.email_config: Optional[Dict] = None
        self.webhook_url: Optional[str] = None
        self.custom_callbacks: List[Callable[[Alert], None]] = []

    def configure_email(
        self,
        smtp_server: str,
        smtp_port: int,
        username: str,
        password: str,
        from_addr: str,
        to_addrs: List[str]
    ):
        """
        Configure email notifications.

        Args:
            smtp_server: SMTP server hostname
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            from_addr: Sender email address
            to_addrs: List of recipient email addresses
        """
        self.email_config = {
            'server': smtp_server,
            'port': smtp_port,
            'username': username,
            'password': password,
            'from': from_addr,
            'to': to_addrs
        }
        logger.info(f"Email notifications configured: {smtp_server}")

    def configure_webhook(self, url: str):
        """
        Configure webhook notifications.

        Args:
            url: URL to POST alert data to
        """
        self.webhook_url = url
        logger.info(f"Webhook notifications configured: {url}")

    def add_callback(self, callback: Callable[[Alert], None]):
        """
        Add a custom callback function for notifications.

        Args:
            callback: Function that takes an Alert and handles notification
        """
        self.custom_callbacks.append(callback)
        logger.info("Custom notification callback added")

    def notify(self, alert: Alert) -> bool:
        """
        Send notifications for an alert.

        Args:
            alert: The alert to notify about

        Returns:
            True if at least one notification was sent successfully
        """
        success = False

        # Always log the alert
        severity_method = getattr(logger, alert.threshold.severity, logger.warning)
        severity_method(
            f"ALERT [{alert.threshold.severity.upper()}] Site {alert.threshold.site_id}: "
            f"{alert.threshold.parameter} {alert.threshold.condition} {alert.threshold.value} "
            f"(current: {alert.current_value:.2f}) - {alert.threshold.message}"
        )
        success = True

        # Email notification
        if self.email_config:
            try:
                self._send_email(alert)
                success = True
            except Exception as e:
                logger.error(f"Email notification failed: {e}")

        # Webhook notification
        if self.webhook_url:
            try:
                self._send_webhook(alert)
                success = True
            except Exception as e:
                logger.error(f"Webhook notification failed: {e}")

        # Custom callbacks
        for callback in self.custom_callbacks:
            try:
                callback(alert)
                success = True
            except Exception as e:
                logger.error(f"Custom callback failed: {e}")

        return success

    def _send_email(self, alert: Alert):
        """Send email notification."""
        if not self.email_config:
            return

        subject = f"[{alert.threshold.severity.upper()}] Hydrology Alert - Site {alert.threshold.site_id}"
        body = f"""
Hydrology Alert Triggered
========================

Site ID: {alert.threshold.site_id}
Parameter: {alert.threshold.parameter}
Condition: {alert.threshold.condition} {alert.threshold.value}
Current Value: {alert.current_value:.2f}
Severity: {alert.threshold.severity}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

Message: {alert.threshold.message}

---
This is an automated alert from the Hydrology Analysis System.
"""

        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = self.email_config['from']
        msg['To'] = ', '.join(self.email_config['to'])

        with smtplib.SMTP(self.email_config['server'], self.email_config['port']) as server:
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            server.send_message(msg)

        logger.info(f"Email notification sent for alert {alert.alert_id}")

    def _send_webhook(self, alert: Alert):
        """Send webhook notification."""
        if not self.webhook_url:
            return

        response = requests.post(
            self.webhook_url,
            json=alert.to_dict(),
            timeout=10
        )
        response.raise_for_status()
        logger.info(f"Webhook notification sent for alert {alert.alert_id}")


class AlertMonitor:
    """
    Main alert monitoring system.

    Manages thresholds, checks conditions, and triggers notifications.

    Example:
        >>> monitor = AlertMonitor()
        >>> monitor.add_threshold(AlertThreshold(
        ...     site_id='12422500',
        ...     parameter='discharge',
        ...     condition='above',
        ...     value=10000,
        ...     severity='warning'
        ... ))
        >>> alerts = monitor.check_all_sites()
        >>> for alert in alerts:
        ...     print(f"Alert: {alert.threshold.message}")
    """

    def __init__(self, history_file: Optional[Path] = None):
        """
        Initialize the alert monitor.

        Args:
            history_file: Path to save alert history (default: outputs/alerts/history.json)
        """
        self.thresholds: List[AlertThreshold] = []
        self.alert_history: List[Alert] = []
        self.last_check: Dict[str, datetime] = {}  # site_id -> last check time
        self.last_values: Dict[str, float] = {}  # site_id -> last value
        self.notifier = AlertNotifier()

        # Setup history file
        if history_file is None:
            alerts_dir = OUTPUT_DIR / 'alerts'
            ensure_dir(alerts_dir)
            history_file = alerts_dir / 'history.json'
        self.history_file = history_file

        # Load existing history
        self._load_history()

    def add_threshold(self, threshold: AlertThreshold):
        """
        Add an alert threshold to monitor.

        Args:
            threshold: AlertThreshold to add
        """
        self.thresholds.append(threshold)
        logger.info(
            f"Added threshold: {threshold.site_id} {threshold.parameter} "
            f"{threshold.condition} {threshold.value}"
        )

    def remove_threshold(self, site_id: str, parameter: str):
        """
        Remove thresholds for a site/parameter combination.

        Args:
            site_id: Site to remove thresholds for
            parameter: Parameter to remove thresholds for
        """
        self.thresholds = [
            t for t in self.thresholds
            if not (t.site_id == site_id and t.parameter == parameter)
        ]

    def check_site(self, site_id: str, use_instantaneous: bool = False) -> List[Alert]:
        """
        Check all thresholds for a specific site.

        Args:
            site_id: USGS site ID to check
            use_instantaneous: If True, fetch instantaneous values instead of daily

        Returns:
            List of triggered alerts
        """
        alerts = []
        site_thresholds = [t for t in self.thresholds if t.site_id == site_id and t.enabled]

        if not site_thresholds:
            return alerts

        # Group thresholds by param_code to minimize API calls
        param_codes = set(t.param_code for t in site_thresholds)

        for param_code in param_codes:
            try:
                # Fetch current data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=1)

                if use_instantaneous:
                    df = fetch_instantaneous_values(
                        site_id, param_code,
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d')
                    )
                else:
                    df = fetch_daily_values(
                        site_id, param_code,
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d')
                    )

                if df is None or df.empty:
                    logger.warning(f"No data available for {site_id} param {param_code}")
                    continue

                # Get current value (most recent)
                current_value = df['value'].iloc[-1]
                previous_value = df['value'].iloc[-2] if len(df) > 1 else None

                # Store for rate calculations
                cache_key = f"{site_id}_{param_code}"
                self.last_values[cache_key] = current_value

                # Check each threshold for this param
                param_thresholds = [t for t in site_thresholds if t.param_code == param_code]

                for threshold in param_thresholds:
                    if self._check_threshold(threshold, current_value, previous_value):
                        # Check cooldown
                        if self._in_cooldown(threshold):
                            continue

                        alert = Alert(
                            threshold=threshold,
                            current_value=current_value,
                            previous_value=previous_value,
                            timestamp=datetime.now()
                        )

                        # Send notification
                        if self.notifier.notify(alert):
                            alert.notified = True

                        alerts.append(alert)
                        self.alert_history.append(alert)
                        self.last_check[f"{site_id}_{threshold.condition}_{threshold.value}"] = datetime.now()

            except Exception as e:
                logger.error(f"Error checking site {site_id} param {param_code}: {e}")

        # Save updated history
        self._save_history()

        return alerts

    def check_all_sites(self, use_instantaneous: bool = False) -> List[Alert]:
        """
        Check all configured thresholds for all sites.

        Args:
            use_instantaneous: If True, fetch instantaneous values

        Returns:
            List of all triggered alerts
        """
        all_alerts = []
        site_ids = set(t.site_id for t in self.thresholds if t.enabled)

        for site_id in site_ids:
            try:
                alerts = self.check_site(site_id, use_instantaneous)
                all_alerts.extend(alerts)
            except Exception as e:
                logger.error(f"Error checking site {site_id}: {e}")

        return all_alerts

    def _check_threshold(
        self,
        threshold: AlertThreshold,
        current: float,
        previous: Optional[float]
    ) -> bool:
        """Check if a threshold condition is met."""
        condition = AlertCondition(threshold.condition)

        if condition == AlertCondition.ABOVE:
            return current > threshold.value
        elif condition == AlertCondition.BELOW:
            return current < threshold.value
        elif condition == AlertCondition.EQUALS:
            return abs(current - threshold.value) < 0.01
        elif condition == AlertCondition.RATE_INCREASE:
            if previous is None:
                return False
            rate = current - previous
            return rate > threshold.value
        elif condition == AlertCondition.RATE_DECREASE:
            if previous is None:
                return False
            rate = current - previous
            return rate < -threshold.value

        return False

    def _in_cooldown(self, threshold: AlertThreshold) -> bool:
        """Check if threshold is in cooldown period."""
        key = f"{threshold.site_id}_{threshold.condition}_{threshold.value}"
        if key not in self.last_check:
            return False

        elapsed = datetime.now() - self.last_check[key]
        return elapsed.total_seconds() < threshold.cooldown_minutes * 60

    def _load_history(self):
        """Load alert history from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded {len(data)} alerts from history")
            except Exception as e:
                logger.warning(f"Could not load alert history: {e}")

    def _save_history(self):
        """Save alert history to file."""
        try:
            # Keep only last 1000 alerts
            recent_alerts = self.alert_history[-1000:]
            data = [a.to_dict() for a in recent_alerts]

            with open(self.history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save alert history: {e}")

    def get_active_alerts(self, hours: int = 24) -> List[Alert]:
        """
        Get alerts from the last N hours.

        Args:
            hours: Number of hours to look back

        Returns:
            List of recent alerts
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        return [a for a in self.alert_history if a.timestamp > cutoff]

    def acknowledge_alert(self, alert_id: str):
        """
        Mark an alert as acknowledged.

        Args:
            alert_id: ID of the alert to acknowledge
        """
        for alert in self.alert_history:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                self._save_history()
                return

    def get_site_status(self, site_id: str) -> Dict[str, Any]:
        """
        Get current monitoring status for a site.

        Args:
            site_id: Site to get status for

        Returns:
            Dict with current values, thresholds, and recent alerts
        """
        site_thresholds = [t for t in self.thresholds if t.site_id == site_id]
        site_alerts = [a for a in self.alert_history if a.threshold.site_id == site_id]
        recent_alerts = [a for a in site_alerts if a.timestamp > datetime.now() - timedelta(hours=24)]

        return {
            'site_id': site_id,
            'thresholds': [asdict(t) for t in site_thresholds],
            'total_alerts': len(site_alerts),
            'recent_alerts_24h': len(recent_alerts),
            'last_values': {k: v for k, v in self.last_values.items() if k.startswith(site_id)}
        }


# Convenience function for quick threshold setup
def create_flood_alert(
    site_id: str,
    flood_stage: float,
    action_stage: Optional[float] = None,
    major_flood_stage: Optional[float] = None
) -> List[AlertThreshold]:
    """
    Create standard flood alert thresholds for a site.

    Args:
        site_id: USGS site ID
        flood_stage: Flood stage in feet
        action_stage: Action stage in feet (optional)
        major_flood_stage: Major flood stage in feet (optional)

    Returns:
        List of AlertThreshold objects for the flood stages
    """
    thresholds = []

    if action_stage:
        thresholds.append(AlertThreshold(
            site_id=site_id,
            parameter='stage',
            condition='above',
            value=action_stage,
            severity='info',
            message=f'Action stage reached at site {site_id}'
        ))

    thresholds.append(AlertThreshold(
        site_id=site_id,
        parameter='stage',
        condition='above',
        value=flood_stage,
        severity='warning',
        message=f'Flood stage reached at site {site_id}'
    ))

    if major_flood_stage:
        thresholds.append(AlertThreshold(
            site_id=site_id,
            parameter='stage',
            condition='above',
            value=major_flood_stage,
            severity='critical',
            message=f'MAJOR FLOOD stage reached at site {site_id}'
        ))

    return thresholds


def create_low_flow_alert(
    site_id: str,
    low_flow_threshold: float,
    critical_flow_threshold: Optional[float] = None
) -> List[AlertThreshold]:
    """
    Create low flow alert thresholds for a site.

    Args:
        site_id: USGS site ID
        low_flow_threshold: Low flow threshold in cfs
        critical_flow_threshold: Critical low flow threshold in cfs (optional)

    Returns:
        List of AlertThreshold objects for low flow conditions
    """
    thresholds = [
        AlertThreshold(
            site_id=site_id,
            parameter='discharge',
            condition='below',
            value=low_flow_threshold,
            severity='warning',
            message=f'Low flow conditions at site {site_id}'
        )
    ]

    if critical_flow_threshold:
        thresholds.append(AlertThreshold(
            site_id=site_id,
            parameter='discharge',
            condition='below',
            value=critical_flow_threshold,
            severity='critical',
            message=f'CRITICAL low flow at site {site_id}'
        ))

    return thresholds
