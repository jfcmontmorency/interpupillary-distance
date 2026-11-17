// Mini serveur statique pour le développement : node serve.js [port]
const http = require("http"), fs = require("fs"), path = require("path");
const root = __dirname, port = +process.env.PORT || +process.argv[2] || 8123;
const types = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".png": "image/png", ".jpg": "image/jpeg" };
http.createServer((req, res) => {
  const p = path.join(root, decodeURIComponent(req.url.split("?")[0]) === "/" ? "index.html" : decodeURIComponent(req.url.split("?")[0]));
  if (!p.startsWith(root)) { res.writeHead(403); return res.end(); }
  fs.readFile(p, (err, data) => {
    if (err) { res.writeHead(404); return res.end("Not found"); }
    res.writeHead(200, { "Content-Type": (types[path.extname(p)] || "application/octet-stream") + "; charset=utf-8", "Cache-Control": "no-store" });
    res.end(data);
  });
}).listen(port, () => console.log(`http://localhost:${port}`));
