import React, { useEffect, useState } from "react";
import { RxHamburgerMenu } from "react-icons/rx";
import { IoIosSearch } from "react-icons/io";
import { CiCircleQuestion } from "react-icons/ci";
import { IoIosSettings } from "react-icons/io";
import { TbGridDots } from "react-icons/tb";
import Avatar from "react-avatar";
import { useDispatch, useSelector } from "react-redux";
import { setSearchText, setAuthUser } from "../redux/appSlice";
import axios from "axios";
import toast from "react-hot-toast";
import { useNavigate } from "react-router-dom";

const Navbar = () => {
  const [text, setText] = useState("");
  const { user } = useSelector((s) => s.app);
  const dispatch = useDispatch();
  const navigate = useNavigate();

  useEffect(() => {
    dispatch(setSearchText(text));
    // eslint-disable-next-line
  }, [text]);

  const logoutHandler = async () => {
    try {
      const res = await axios.get("/api/v1/user/logout", { withCredentials: true });
      toast.success(res.data.message);
      dispatch(setAuthUser(null));
      navigate("/login");
    } catch (err) {
      console.error(err);
      toast.error("Logout failed");
    }
  };

  return (
    <header className="flex items-center justify-between px-6 py-3 bg-white/60 backdrop-blur-sm border-b">
      <div className="flex items-center gap-4">
        <button className="p-2 rounded-lg hover:bg-gray-100 transition">
          <RxHamburgerMenu size={20} />
        </button>

        <div className="flex items-center gap-3">
          <img src="https://mailmeteor.com/logos/assets/PNG/Gmail_Logo_512px.png" alt="logo" className="w-8" />
          <span className="text-lg font-semibold text-gray-800">MyMail</span>
        </div>
      </div>

      <div className="flex-1 flex justify-center">
        <div className="w-2/5 bg-gray-100 rounded-full px-4 py-2 flex items-center gap-3 shadow-inner border border-gray-200">
          <IoIosSearch size={18} className="text-gray-500" />
          <input
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Search mail and contacts"
            className="bg-transparent outline-none w-full text-gray-700"
          />
        </div>
      </div>

      <div className="flex items-center gap-4">
       <button
  onClick={() => window.open("http://127.0.0.1:5500/spam-detector%20test/spam-detector%20test/index.html", "_blank")}
  className="px-4 py-2 bg-blue-600 text-white rounded-full font-medium shadow hover:bg-blue-700 transition-all"
>
  Email Classifier
</button>


        <CiCircleQuestion size={20} className="text-gray-600 hover:text-primary cursor-pointer" />
        <IoIosSettings size={20} className="text-gray-600 hover:text-primary cursor-pointer" />
        <TbGridDots size={20} className="text-gray-600 hover:text-primary cursor-pointer" />

        <button onClick={logoutHandler} className="text-sm text-gray-700 hover:text-red-600 underline">
          Logout
        </button>

        <Avatar src={user?.profilePhoto} size="38" round />
      </div>
    </header>
  );
};

export default Navbar;
