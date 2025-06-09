import React, { useState } from 'react';

function App() {
  const [jobDescription, setJobDescription] = useState('');
  const [files, setFiles] = useState([]);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false); // ✅ Added loader state

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!jobDescription || files.length === 0) {
      alert("Please enter a job description and upload at least one PDF.");
      return;
    }

    const formData = new FormData();
    formData.append("job_description", jobDescription); // ✅ Matches Flask
    for (const file of files) {
      formData.append("resumes", file); // ✅ Matches Flask key
    }

    try {
      setLoading(true); // ✅ Start loader
      const res = await fetch("http://localhost:5000/shortlist", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (res.ok) {
        setResults(data);
      } else {
        alert(data.error || "Something went wrong.");
      }
    } catch (error) {
      console.error("Error submitting form:", error);
    } finally {
      setLoading(false); // ✅ End loader
    }
  };

  return (
    <div style={{
      backgroundColor: "#121212",
      color: "#f0f0f0",
      minHeight: "100vh",
      padding: "2rem",
      fontFamily: "Segoe UI, sans-serif"
    }}>
      <h2 style={{ color: "#90caf9" }}>Resume Shortlister AI</h2>

      <form onSubmit={handleSubmit}>
        <textarea
          rows="4"
          placeholder="Enter job description"
          value={jobDescription}
          onChange={(e) => setJobDescription(e.target.value)}
          style={{
            width: "100%",
            marginBottom: "1rem",
            padding: "1rem",
            backgroundColor: "#1e1e1e",
            color: "#fff",
            border: "1px solid #333",
            borderRadius: "6px"
          }}
        />
        <input type="file" accept=".pdf,.docx" multiple onChange={(e) => setFiles([...e.target.files])}
          style={{ marginBottom: "1rem" }} />

        <br />
        <button type="submit" style={{
          padding: "0.5rem 1rem",
          backgroundColor: "#1976d2",
          color: "#fff",
          border: "none",
          borderRadius: "6px",
          cursor: "pointer"
        }}>
          Shortlist Resumes
        </button>
      </form>

      {loading && (
        <p style={{ color: "#bb86fc", marginTop: "1rem" }}>
          ⏳ Processing resumes... Please wait.
        </p>
      )}

      <hr style={{ margin: "2rem 0", borderColor: "#444" }} />

      {results.length > 0 && (
        <div>
          <h3 style={{ color: "#bb86fc" }}>Top Matches</h3>
          {[...results]
            .sort((a, b) => b.score - a.score)
            .slice(0, 3)
            .map((res, i) => (
              <div
                key={i}
                style={{
                  backgroundColor: "#1e1e1e",
                  padding: "1rem",
                  borderRadius: "8px",
                  marginBottom: "1rem",
                  boxShadow: "0 0 10px rgba(255,255,255,0.05)"
                }}
              >
                <h4 style={{ color: "#03dac6" }}>
                  {res.metadata?.name || res.filename} (Score: {Math.round(res.score)}%)
                </h4>
                <p>{typeof res.feedback === 'string' ? res.feedback : JSON.stringify(res.feedback)}</p>

                <a
                  href={`/uploads/${res.filename}`}
                  download
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{
                    display: "inline-block",
                    marginTop: "0.5rem",
                    color: "#90caf9",
                    textDecoration: "none",
                    fontWeight: "bold"
                  }}
                >
                  ⬇️ Download Resume
                </a>
              </div>
            ))}
        </div>
      )}
    </div>
  );
}

export default App;
