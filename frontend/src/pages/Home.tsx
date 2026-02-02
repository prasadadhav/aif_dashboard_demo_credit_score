import React from "react";
import { ChartBlock } from "../components/runtime/ChartBlock";
import { TableBlock } from "../components/runtime/TableBlock";
import { MetricCardBlock } from "../components/runtime/MetricCardBlock";

const Home: React.FC = () => {
  return (
    <div id="7gx6FKvT6NwQwvkM">
    <div id="iofcf" className="gjs-row" style={{"width": "100%", "height": "auto", "padding": "0", "margin": "0", "position": "static", "textAlign": "left", "zIndex": 0, "--chart-color-palette": "default", "display": "flex", "paddingTop": "10px", "paddingRight": "10px", "paddingBottom": "10px", "paddingLeft": "10px", "flexWrap": "wrap"}}>
      <div id="if369" className="gjs-cell" style={{"height": "auto", "padding": "0", "margin": "0", "position": "static", "textAlign": "left", "zIndex": 0, "--chart-color-palette": "default", "flex": "1 1 calc(33.333% - 20px)", "minWidth": "250px"}}>
        <nav id="iz4aj" style={{"background": "linear-gradient(135deg, #4b3c82 0%, #5a3d91 100%)", "color": "white", "padding": "15px 30px", "display": "flex", "justifyContent": "space-between", "alignItems": "center", "fontFamily": "Arial, sans-serif", "--chart-color-palette": "default"}}>
          <p id="i8eko" style={{"fontSize": "24px", "fontWeight": "bold", "textAlign": "center", "--chart-color-palette": "default"}}>{"AI Factories Dashboard"}</p>
          <div id="iywdg" style={{"display": "flex", "gap": "30px", "--chart-color-palette": "default"}}>
            <a id="igwk8" style={{"color": "white", "textDecoration": "none", "--chart-color-palette": "default"}} href="/">{"Home"}</a>
            <a id="ibki7" style={{"color": "white", "textDecoration": "none", "--chart-color-palette": "default"}} href="/about">{"About"}</a>
          </div>
        </nav>
        <section id="container_section" className="bdg-sect">
          <h1 id="h1" className="heading" style={{"width": "auto", "height": "auto", "padding": "0", "margin": "0", "position": "static", "textAlign": "center", "zIndex": 0, "--chart-color-palette": "default"}}>{"Credit Score rating based on ML model"}</h1>
          <p id="p" className="paragraph" style={{"width": "auto", "height": "auto", "padding": "0", "margin": "0", "position": "static", "textAlign": "center", "zIndex": 0, "--chart-color-palette": "default"}}>{"The ML model is trained on different features & different parameters."}</p>
        </section>
        <ChartBlock id="isv553" styles={{"width": "100%", "minHeight": "400px", "--chart-color-palette": "default"}} chartType="line-chart" title="Model Performance" color="#4CAF50" chart={{"lineWidth": 2, "curveType": "monotone", "showGrid": true, "showLegend": true, "showTooltip": true, "animate": true, "legendPosition": "top", "gridColor": "#e0e0e0", "dotSize": 5}} series={[{"name": "Series_1", "label": "Series 1", "color": "#f51d1d"}, {"name": "Series_2", "label": "Series 2", "color": "#1711f2"}]} />
        <div id="idd2yg" className="gjs-row" style={{"display": "flex", "padding": "10px", "width": "100%", "--chart-color-palette": "default", "paddingTop": "10px", "paddingRight": "10px", "paddingBottom": "10px", "paddingLeft": "10px", "flexWrap": "wrap"}}>
          <div id="ixa3hr" className="gjs-cell" style={{"height": "auto", "--chart-color-palette": "default", "flex": "1 1 calc(33.333% - 20px)", "minWidth": "250px"}}>
            <div id="iy202f" className="gjs-row" style={{"width": "100%", "height": "auto", "padding": "0", "margin": "0", "position": "static", "textAlign": "left", "zIndex": 0, "--chart-color-palette": "default", "display": "flex", "paddingTop": "10px", "paddingRight": "10px", "paddingBottom": "10px", "paddingLeft": "10px", "flexWrap": "wrap"}}>
              <div id="container_cell" className="gjs-cell" style={{"height": "auto", "padding": "0", "margin": "0", "position": "static", "textAlign": "left", "zIndex": 0, "--chart-color-palette": "default", "flex": "1 1 calc(33.333% - 20px)", "minWidth": "250px"}}>
                <ChartBlock id="iicw2x" styles={{"width": "100%", "minHeight": "400px", "--chart-color-palette": "default"}} chartType="line-chart" title="Model Drift: Jensenshannon" color="#4CAF50" chart={{"lineWidth": 2, "curveType": "monotone", "showGrid": true, "showLegend": true, "showTooltip": true, "animate": true, "legendPosition": "top", "gridColor": "#e0e0e0", "dotSize": 5}} series={[{"name": "Series_1", "label": "Series 1", "color": "#f51d1d"}, {"name": "Series_2", "label": "Series 2", "color": "#1711f2"}]} />
              </div>
              <div id="container_cell_2" className="gjs-cell" style={{"height": "auto", "padding": "0", "margin": "0", "position": "static", "textAlign": "left", "zIndex": 0, "--chart-color-palette": "default", "flex": "1 1 calc(33.333% - 20px)", "minWidth": "250px"}}>
                <ChartBlock id="iyyzv8" styles={{"width": "100%", "minHeight": "400px", "--chart-color-palette": "default"}} chartType="line-chart" title="Model Drift: Wasserstein" color="#4CAF50" chart={{"lineWidth": 2, "curveType": "monotone", "showGrid": true, "showLegend": true, "showTooltip": true, "animate": true, "legendPosition": "top", "gridColor": "#e0e0e0", "dotSize": 5}} series={[{"name": "Series_1", "label": "Series 1", "color": "#f51d1d"}, {"name": "Series_2", "label": "Series 2", "color": "#1711f2"}]} />
              </div>
            </div>
          </div>
        </div>
        <TableBlock id="iibsn7" styles={{"width": "100%", "minHeight": "400px", "--chart-color-palette": "default"}} title="Data anamoly" options={{"showHeader": true, "stripedRows": false, "showPagination": true, "rowsPerPage": 5, "actionButtons": true, "columns": [{"label": "Feature Type", "column_type": "field", "field": "feature_type", "type": "str", "required": true}, {"label": "Min Value", "column_type": "field", "field": "min_value", "type": "float", "required": true}, {"label": "Max Value", "column_type": "field", "field": "max_value", "type": "float", "required": true}, {"label": "Name", "column_type": "field", "field": "name", "type": "str", "required": true}, {"label": "Description", "column_type": "field", "field": "description", "type": "str", "required": true}, {"label": "Date", "column_type": "lookup", "path": "date", "entity": "Datashape", "field": "accepted_target_values", "type": "str", "required": true}, {"label": "Features", "column_type": "lookup", "path": "features", "entity": "Datashape", "field": "accepted_target_values", "type": "str", "required": true}, {"label": "Project", "column_type": "lookup", "path": "project", "entity": "Project", "field": "name", "type": "str", "required": false}, {"label": "Evalu", "column_type": "lookup", "path": "evalu", "entity": "Evaluation", "field": "status", "type": "list", "required": false}, {"label": "Eval", "column_type": "lookup", "path": "eval", "entity": "Evaluation", "field": "status", "type": "list", "required": false}]}} dataBinding={{"entity": "Feature", "endpoint": "/feature/"}} />
        <div id="i75sw8" className="dashboard-container" style={{"padding": "20px", "background": "linear-gradient(135deg, rgb(75, 60, 130) 0%, rgb(90, 61, 145) 100%)", "--chart-color-palette": "default"}}>
          <div id="icu0vc" style={{"maxWidth": "1400px", "margin": "0 auto", "--chart-color-palette": "default"}}>
            <div id="iip80d" style={{"marginBottom": "30px"}}>
              <h1 id="ij8249" style={{"color": "white", "margin": "0 0 10px 0", "fontSize": "32px", "--chart-color-palette": "default"}}>{"Analytics Dashboard"}</h1>
              <p id="iqomwa" style={{"color": "rgba(255,255,255,0.8)", "margin": "0", "--chart-color-palette": "default"}}>{"Key insights and metrics"}</p>
            </div>
            <div id="im3mgi" className="kpi-row" style={{"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(250px, 1fr))", "gap": "20px", "marginBottom": "30px", "--chart-color-palette": "default"}}>
              <MetricCardBlock id="iy7zsy" styles={{"width": "100%", "minHeight": "140px", "--chart-color-palette": "default"}} metric={{"metricTitle": "Number of Features", "format": "number", "valueColor": "#2c3e50", "valueSize": 32, "showTrend": true, "positiveColor": "#27ae60", "negativeColor": "#e74c3c", "value": 0, "trend": 12}} dataBinding={{"entity": "Model", "endpoint": "/model/", "data_field": "source"}} />
              <MetricCardBlock id="ip3i4k" styles={{"width": "100%", "minHeight": "140px", "--chart-color-palette": "default"}} metric={{"metricTitle": "Total Models evaluated", "format": "number", "valueColor": "#2c3e50", "valueSize": 32, "showTrend": true, "positiveColor": "#27ae60", "negativeColor": "#e74c3c", "value": 0, "trend": 12}} dataBinding={{"entity": "Model", "endpoint": "/model/", "data_field": "pid"}} />
              <MetricCardBlock id="ilcwkv" styles={{"width": "100%", "minHeight": "140px", "--chart-color-palette": "default"}} metric={{"metricTitle": "No. of Metrics", "format": "number", "valueColor": "#2c3e50", "valueSize": 32, "showTrend": true, "positiveColor": "#27ae60", "negativeColor": "#e74c3c", "value": 0, "trend": 12}} dataBinding={{"entity": "Measure", "endpoint": "/measure/", "data_field": "Metric"}} />
            </div>
            <div id="i34fiw" className="charts-row" style={{"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(500px, 1fr))", "gap": "20px", "marginBottom": "20px", "--chart-color-palette": "default"}} />
          </div>
        </div>
        <TableBlock id="ioj8df" styles={{"width": "100%", "minHeight": "400px", "--chart-color-palette": "default"}} title="Comments" options={{"showHeader": true, "stripedRows": false, "showPagination": true, "rowsPerPage": 5, "actionButtons": true, "columns": [{"label": "Name", "column_type": "field", "field": "Name", "type": "str", "required": true}, {"label": "Comments", "column_type": "field", "field": "Comments", "type": "str", "required": true}, {"label": "TimeStamp", "column_type": "field", "field": "TimeStamp", "type": "datetime", "required": true}]}} dataBinding={{"entity": "Comments", "endpoint": "/comments/"}} />
        <footer id="iru6jv" style={{"background": "linear-gradient(135deg, #4b3c82 0%, #5a3d91 100%)", "color": "white", "padding": "40px 20px", "fontFamily": "Arial, sans-serif", "--chart-color-palette": "default"}}>
          <div id="i1lq9c" style={{"maxWidth": "1200px", "margin": "0 auto", "display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(250px, 1fr))", "gap": "30px", "--chart-color-palette": "default"}}>
            <div id="Component">
              <h4 id="i6xtqz" style={{"marginTop": "0"}}>{"About BESSER"}</h4>
              <p id="i2wbui" style={{"opacity": "0.8", "lineHeight": "1.6", "--chart-color-palette": "default"}}>{"BESSER is a low-code platform for building smarter software faster. Empower your development with our dashboard generator and modeling tools."}</p>
            </div>
            <div id="Component_2">
              <h4 id="imf0qz" style={{"marginTop": "0"}}>{"Quick Links"}</h4>
              <nav id="if21il" style={{"listStyle": "none", "padding": "0", "opacity": "0.8", "--chart-color-palette": "default"}}>
                <ul style={{"listStyle": "none", "padding": 0, "margin": 0}}>
                  <li style={{"display": "inline-block", "marginRight": "15px"}}><a href="/about">{"About"}</a></li>
                </ul>
              </nav>
            </div>
            <div id="Component_3">
              <h4 id="ij7xsi" style={{"marginTop": "0"}}>{"Contact"}</h4>
              <p id="iw5bdf" style={{"opacity": "0.8", "--chart-color-palette": "default"}}>{"Email: info@besser-pearl.org\nPhone: (123) 456-7890"}</p>
            </div>
          </div>
          <p id="itwa7j" style={{"textAlign": "center", "marginTop": "30px", "paddingTop": "20px", "borderTop": "1px solid rgba(255,255,255,0.1)", "opacity": "0.7", "--chart-color-palette": "default"}}>{"© 2025 BESSER. All rights reserved."}</p>
        </footer>
      </div>
    </div>    </div>
  );
};

export default Home;
