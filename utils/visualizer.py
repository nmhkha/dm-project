import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

class Visualizer:
    """Handles all visualization tasks for the data cleaning platform."""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
    
    def plot_missing_values_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create a heatmap showing missing values pattern."""
        missing_data = df.isnull()
        
        fig = go.Figure(data=go.Heatmap(
            z=missing_data.values,
            x=missing_data.columns,
            y=list(range(len(missing_data))),
            colorscale=[[0, 'lightblue'], [1, 'red']],
            showscale=True,
            colorbar=dict(title="Missing Values", tickvals=[0, 1], ticktext=["Present", "Missing"])
        ))
        
        fig.update_layout(
            title="Missing Values Pattern",
            xaxis_title="Columns",
            yaxis_title="Row Index",
            height=400
        )
        
        return fig
    
    def plot_missing_values_summary(self, df: pd.DataFrame) -> go.Figure:
        """Create a bar chart showing missing values count per column."""
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Missing Values Count', 'Missing Values Percentage'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Count plot
        fig.add_trace(
            go.Bar(x=missing_counts.index, y=missing_counts.values, 
                   name="Count", marker_color='lightcoral'),
            row=1, col=1
        )
        
        # Percentage plot
        fig.add_trace(
            go.Bar(x=missing_percentages.index, y=missing_percentages.values,
                   name="Percentage", marker_color='lightblue'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        fig.update_xaxes(tickangle=45)
        
        return fig
    
    def plot_data_distribution(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> go.Figure:
        """Create distribution plots for numeric columns."""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        n_cols = min(len(columns), 3)
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=columns[:n_rows * n_cols]
        )
        
        for i, col in enumerate(columns[:n_rows * n_cols]):
            row = (i // n_cols) + 1
            col_pos = (i % n_cols) + 1
            
            fig.add_trace(
                go.Histogram(x=df[col].dropna(), name=col, nbinsx=30),
                row=row, col=col_pos
            )
        
        fig.update_layout(height=300 * n_rows, showlegend=False)
        return fig
    
    def plot_outliers_boxplot(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> go.Figure:
        """Create box plots to visualize outliers."""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        fig = go.Figure()
        
        for i, col in enumerate(columns[:6]):  # Limit to 6 columns for readability
            fig.add_trace(go.Box(
                y=df[col].dropna(),
                name=col,
                boxpoints='outliers',
                marker_color=self.color_palette[i % len(self.color_palette)]
            ))
        
        fig.update_layout(
            title="Outliers Detection (Box Plots)",
            yaxis_title="Values",
            height=500
        )
        
        return fig
    
    def plot_categorical_distribution(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> go.Figure:
        """Create bar plots for categorical columns."""
        if columns is None:
            columns = df.select_dtypes(include=['object']).columns.tolist()
        
        n_cols = min(len(columns), 2)
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=columns[:n_rows * n_cols]
        )
        
        for i, col in enumerate(columns[:n_rows * n_cols]):
            row = (i // n_cols) + 1
            col_pos = (i % n_cols) + 1
            
            value_counts = df[col].value_counts().head(10)
            
            fig.add_trace(
                go.Bar(x=value_counts.index, y=value_counts.values, name=col),
                row=row, col=col_pos
            )
        
        fig.update_layout(height=300 * n_rows, showlegend=False)
        fig.update_xaxes(tickangle=45)
        
        return fig
    
    def plot_correlation_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create correlation heatmap for numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return None
        
        corr_matrix = df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Correlation Heatmap",
            height=500,
            width=500
        )
        
        return fig
    
    def plot_data_quality_comparison(self, original_scores: Dict, cleaned_scores: Dict) -> go.Figure:
        """Compare data quality scores before and after cleaning."""
        metrics = list(original_scores.keys())
        original_values = list(original_scores.values())
        cleaned_values = list(cleaned_scores.values())
        
        fig = go.Figure(data=[
            go.Bar(name='Before Cleaning', x=metrics, y=original_values, marker_color='lightcoral'),
            go.Bar(name='After Cleaning', x=metrics, y=cleaned_values, marker_color='lightgreen')
        ])
        
        fig.update_layout(
            title="Data Quality Score Comparison",
            xaxis_title="Quality Metrics",
            yaxis_title="Score (0-100)",
            barmode='group',
            height=400
        )
        
        return fig
    
    def plot_before_after_comparison(self, original: pd.DataFrame, modified: pd.DataFrame, 
                                   metric: str = 'missing_values') -> go.Figure:
        """Create before/after comparison plots."""
        if metric == 'missing_values':
            original_missing = original.isnull().sum()
            modified_missing = modified.isnull().sum()
            
            fig = go.Figure(data=[
                go.Bar(name='Original', x=original_missing.index, y=original_missing.values, 
                       marker_color='lightcoral'),
                go.Bar(name='Modified', x=modified_missing.index, y=modified_missing.values,
                       marker_color='lightblue')
            ])
            
            fig.update_layout(
                title="Missing Values: Before vs After",
                xaxis_title="Columns",
                yaxis_title="Missing Values Count",
                barmode='group',
                height=400
            )
            
        elif metric == 'data_types':
            # Show data type changes
            type_changes = {}
            for col in original.columns:
                if col in modified.columns:
                    orig_type = str(original[col].dtype)
                    mod_type = str(modified[col].dtype)
                    if orig_type != mod_type:
                        type_changes[col] = f"{orig_type} → {mod_type}"
            
            if type_changes:
                fig = go.Figure(data=go.Table(
                    header=dict(values=['Column', 'Type Change']),
                    cells=dict(values=[list(type_changes.keys()), list(type_changes.values())])
                ))
                fig.update_layout(title="Data Type Changes", height=300)
            else:
                fig = go.Figure()
                fig.add_annotation(text="No data type changes detected", 
                                 xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        return fig
    
    def create_data_overview_plots(self, df: pd.DataFrame) -> Dict[str, go.Figure]:
        """Create a comprehensive set of overview plots."""
        plots = {}
        
        # Missing values
        if df.isnull().sum().sum() > 0:
            plots['missing_heatmap'] = self.plot_missing_values_heatmap(df)
            plots['missing_summary'] = self.plot_missing_values_summary(df)
        
        # Distributions
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            plots['distributions'] = self.plot_data_distribution(df)
            plots['outliers'] = self.plot_outliers_boxplot(df)
            
            if len(numeric_cols) > 1:
                plots['correlation'] = self.plot_correlation_heatmap(df)
        
        # Categorical
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            plots['categorical'] = self.plot_categorical_distribution(df)
        
        return plots
