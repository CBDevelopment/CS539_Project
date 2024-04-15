
import "./componentStyles.css";

export default function Navbar() {
    return (
        <nav className="navbar">
            <div className="navbar-brand">
                <a href="#home" className="navbar-logo">Image Locator</a>
            </div>
            <ul className="navbar-nav">
                <li className="nav-item">
                    <a href="#about" className="nav-link">About</a>
                </li>
            </ul>
        </nav>
    );
}