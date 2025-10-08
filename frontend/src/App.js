import React, { useMemo, useState, useEffect } from "react";
import {
  Layout,
  Typography,
  Upload,
  Button,
  Table,
  Card,
  Row,
  Col,
  Statistic,
  message,
  Input,
  Space,
  Alert,
  Select,
} from "antd";
import { UploadOutlined } from "@ant-design/icons";
import axios from "axios";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Legend,
} from "recharts";
import { Brush } from "recharts";


const { Header, Content, Footer } = Layout;
const { Title, Text } = Typography;
const { Option } = Select;

const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:8000";

function App() {
  const [health, setHealth] = useState(null);
  const [rows, setRows] = useState([]);
  const [kpis, setKpis] = useState(null);
  const [loadingUpload, setLoadingUpload] = useState(false); // separate loading
  const [loadingSingle, setLoadingSingle] = useState(false); // separate loading
  const [singleResponse, setSingleResponse] = useState(null);
  const [serverError, setServerError] = useState(null);

  useEffect(() => {
    axios
      .get(`${API_BASE}/health`)
      .then((r) => setHealth(r.data))
      .catch(() => setHealth(null));
  }, []);

  const columns = useMemo(
    () => [
      { title: "Index", dataIndex: "index", width: 80 },
      { title: "Price (Reco)", dataIndex: "price_recommended" },
      { title: "p_complete (Reco)", dataIndex: "p_complete_recommended" },
      { title: "p_complete (Base)", dataIndex: "p_complete_baseline" },
      { title: "GM %", dataIndex: "gm_pct" },
      { title: "Bound Low", dataIndex: "bound_low" },
      { title: "Bound High", dataIndex: "bound_high" },
     
    ],
    []
  );

  const chartData = useMemo(() => {
    return rows.map((r, i) => ({
      i,
      reco: typeof r.price_recommended === "number" ? r.price_recommended : null,
      gm: typeof r.gm_pct === "number" ? r.gm_pct : null,
      p_reco:
        typeof r.p_complete_recommended === "number" ? r.p_complete_recommended : null,
      p_base:
        typeof r.p_complete_baseline === "number" ? r.p_complete_baseline : null,
    }));
  }, [rows]);

  const normalizeRowsFromServer = (rawRows = []) => {
    return rawRows.map((r, i) => {
      const idx = Number.isFinite(r.index) ? r.index : i;
      return {
        index: idx,
        price_recommended:
          r.price_recommended !== undefined ? Number(r.price_recommended) : undefined,
        p_complete_recommended:
          r.p_complete_recommended !== undefined
            ? Number(r.p_complete_recommended)
            : undefined,
        p_complete_baseline:
          r.p_complete_baseline !== undefined ? Number(r.p_complete_baseline) : undefined,
        gm_pct: r.gm_pct !== undefined ? Number(r.gm_pct) : undefined,
        bound_low: r.bound_low !== undefined ? Number(r.bound_low) : undefined,
        bound_high: r.bound_high !== undefined ? Number(r.bound_high) : undefined,
        error: r.error || null,
      };
    });
  };

  const onUpload = async (file) => {
    setLoadingUpload(true);
    setServerError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const { data } = await axios.post(`${API_BASE}/recommend_batch`, form, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const normalized = normalizeRowsFromServer(data.rows || []);
      setRows(normalized);
      setKpis(data.kpis || null);
      message.success(`Processed ${data.n_rows ?? normalized.length} rows`);
    } catch (e) {
      console.error(e);
      setServerError(
        e?.response?.data?.detail ||
          e?.response?.data ||
          e?.message ||
          "Failed to process CSV"
      );
      message.error("Failed to process CSV");
    } finally {
      setLoadingUpload(false);
    }
    return false;
  };

  const [jsonInput, setJsonInput] = useState({
    Historical_Cost_of_Ride: 250,
    Expected_Ride_Duration: 35,
    Number_of_Riders: 120,
    Number_of_Drivers: 100,
    Vehicle_Type: "Economy",
    Time_of_Booking: "Evening",
    Location_Category: "Urban",
    Customer_Loyalty_Status: "Silver",
    competitor_price: 360,
  });

  const updateField = (field, value) => {
    setJsonInput((prev) => ({ ...prev, [field]: value }));
  };

  const testSingle = async () => {
    setLoadingSingle(true);
    setServerError(null);
    setSingleResponse(null);
    try {
      const payload = { record: jsonInput };
      const { data } = await axios.post(`${API_BASE}/recommend`, payload, {
        headers: { "Content-Type": "application/json" },
      });
      setSingleResponse(data);
      message.success(
        `Reco: ₹${data.price_recommended} | p=${data.p_complete_recommended}`
      );
    } catch (e) {
      console.error(e);
      const detail =
        e?.response?.data?.detail || e?.response?.data || e?.message || "Server error";
      setServerError(detail);
      message.error("Invalid input or server error");
    } finally {
      setLoadingSingle(false);
    }
  };

  return (
    <Layout style={{ minHeight: "100vh" }}>
      <Header
        style={{
          color: "white",
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
        }}
      >
        <Title
          level={3}
          style={{
            color: "white",
            margin: 0,
            fontWeight: "bold",
            textAlign: "center",
          }}
        >
          PriceOptima — Dashboard
        </Title>
      </Header>

      <Content style={{ padding: 24, maxWidth: 1600, margin: "0 auto", minHeight: 600 }}>

        {serverError && <Alert type="error" message={serverError} showIcon closable />}

        {/* Test Single Record Section */}
        <Card
          title="Test Single Record"
          style={{ marginBottom: 24, background: "#fafafa", minWidth: "1200px", }}
        >
          <Space direction="vertical" style={{ width: "100%" }}>
            <label>Historical Cost of Ride</label>
            <Input
              type="number"
              value={jsonInput.Historical_Cost_of_Ride}
              onChange={(e) =>
                updateField("Historical_Cost_of_Ride", Number(e.target.value))
              }
            />

            <label>Expected Ride Duration</label>
            <Input
              type="number"
              value={jsonInput.Expected_Ride_Duration}
              onChange={(e) =>
                updateField("Expected_Ride_Duration", Number(e.target.value))
              }
            />

            <label>Number of Riders</label>
            <Input
              type="number"
              value={jsonInput.Number_of_Riders}
              onChange={(e) => updateField("Number_of_Riders", Number(e.target.value))}
            />

            <label>Number of Drivers</label>
            <Input
              type="number"
              value={jsonInput.Number_of_Drivers}
              onChange={(e) => updateField("Number_of_Drivers", Number(e.target.value))}
            />

           <label>Vehicle Type</label>
<Select
  value={jsonInput.Vehicle_Type}
  onChange={(val) => updateField("Vehicle_Type", val)}
  style={{ width: "100%" }}
>
  <Option value="Economy">Economy</Option>
  <Option value="Premium">Premium</Option>
</Select>


            <label>Time of Booking</label>
            <Select
              value={jsonInput.Time_of_Booking}
              onChange={(val) => updateField("Time_of_Booking", val)}
              style={{ width: "100%" }}
            >
              <Option value="Morning">Morning</Option>
              <Option value="Afternoon">Afternoon</Option>
              <Option value="Evening">Evening</Option>
              <Option value="Night">Night</Option>
            </Select>

            <label>Location Category</label>
            <Select
              value={jsonInput.Location_Category}
              onChange={(val) => updateField("Location_Category", val)}
              style={{ width: "100%" }}
            >
              <Option value="Urban">Urban</Option>
              <Option value="Suburban">Suburban</Option>
              <Option value="Rural">Rural</Option>
            </Select>

            <label>Customer Loyalty Status</label>
            <Select
              value={jsonInput.Customer_Loyalty_Status}
              onChange={(val) => updateField("Customer_Loyalty_Status", val)}
              style={{ width: "100%" }}
            >
              <Option value="Regular">Regular</Option>
              <Option value="Gold">Gold</Option>
              <Option value="Silver">Silver</Option>
            </Select>

            <label>Competitor Price</label>
            <Input
              type="number"
              value={jsonInput.competitor_price}
              onChange={(e) =>
                updateField("competitor_price", Number(e.target.value))
              }
            />

            <Button type="primary" onClick={testSingle} loading={loadingSingle}
            style={{ width: "100%" }} >
              Test Recommend
            </Button>

           {singleResponse && (
  <Alert
    style={{
      marginTop: 12,
      padding: "16px",
      fontSize: "18px",
      fontWeight: "bold",
      textAlign: "center",
      background: "#d9f7be", // light green background
      border: "2px solid #97eb6dff",
      borderRadius: "8px",
      
    }}
    message={
      <span>
        Recommended:&nbsp;
        <span style={{ color: "black", fontWeight: "bold", fontSize: "22px" }}>
          ₹{singleResponse.price_recommended}
        </span>
        ,&nbsp;p=
        <span style={{ color: "black", fontSize: "22px", fontWeight: "bold" }}>
          {singleResponse.p_complete_recommended}
        </span>
      </span>
    }
    type="success"
    showIcon={false}
  />
)}

          </Space>
        </Card>

        {/* Upload CSV section */}
        <Card
          title="Upload CSV for Batch Recommendation"
          style={{ marginBottom: 24, background: "#fafafa" }}
        >
          <Upload beforeUpload={onUpload} showUploadList={false}>
            <Button icon={<UploadOutlined />} loading={loadingUpload}>
              Upload CSV
            </Button>
          </Upload>
        </Card>

        {/* KPI Section */}
        {kpis && (
          <Row gutter={16} style={{ marginBottom: 24 }}>
            {Object.entries(kpis).map(([k, v]) => (
              <Col span={8} key={k}>
                <Card>
                  <Statistic title={k} value={v} precision={2} />
                </Card>
              </Col>
            ))}
          </Row>
        )}

        {/* Table and Charts */}
        {rows.length > 0 && (
          <>
            <Card title="Recommendations Table">
              <Table
                dataSource={rows}
                columns={columns}
                rowKey="index"
                pagination={{ pageSize: 10 }}
              />
            </Card>

            <Row gutter={16} style={{ marginTop: 24 }}>
              <Col span={12}>
                <Card title="Recommended Prices">
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="i" />
                      <YAxis />
                      <Tooltip />
                      <Line type="monotone" dataKey="reco" stroke="#1890ff" />
                      <Brush dataKey="i" height={20} stroke="#8884d8" />
                    </LineChart>
                  </ResponsiveContainer>
                </Card>
              </Col>
              <Col span={12}>
                <Card title="p_complete (Base vs Reco)">
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="i" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="p_base" fill="#8884d8" name="Baseline" />
                      <Bar dataKey="p_reco" fill="#82ca9d" name="Recommended" />
                      <Brush dataKey="i" height={20} stroke="#82ca9d" />

                    </BarChart>
                  </ResponsiveContainer>
                </Card>
              </Col>
            </Row>
          </>
        )}
      </Content>

      <Footer style={{ textAlign: "center" }}>
        Dynamic Pricing Dashboard ©2025
      </Footer>
    </Layout>
  );
}

export default App;
