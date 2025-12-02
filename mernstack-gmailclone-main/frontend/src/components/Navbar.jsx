import React, { useEffect, useState } from 'react'
import { RxHamburgerMenu } from "react-icons/rx";
import { IoIosSearch } from "react-icons/io";
import { CiCircleQuestion } from "react-icons/ci";
import { IoIosSettings } from "react-icons/io";
import { TbGridDots } from "react-icons/tb";
import Avatar from 'react-avatar';
import { useDispatch, useSelector } from 'react-redux';
import { setAuthUser, setSearchText } from '../redux/appSlice';
import axios from 'axios';
import toast from "react-hot-toast";
import { useNavigate } from 'react-router-dom';

const Navbar = () => {
    const [text, setText] = useState("");
    const { user } = useSelector(store => store.app);
    const dispatch = useDispatch();
    const navigate = useNavigate();

    const logoutHandler = async () => {
        try {
            const res = await axios.get('http://localhost:8080/api/v1/user/logout', { withCredentials: true });
            toast.success(res.data.message);
            dispatch(setAuthUser(null));
            navigate("/login");
        } catch (error) {
            console.log(error);
        }
    }

    useEffect(() => {
        dispatch(setSearchText(text));
    }, [text]);

    return (
        <div className='flex items-center justify-between px-4 h-16 bg-white shadow-sm'>

            {/* LEFT SECTION */}
            <div className='flex items-center gap-4'>
                <div className='p-3 hover:bg-gray-200 rounded-full cursor-pointer'>
                    <RxHamburgerMenu size={22} />
                </div>

                <img
                    className='w-8'
                    src="https://mailmeteor.com/logos/assets/PNG/Gmail_Logo_512px.png"
                    alt="logo"
                />

                <h1 className='text-2xl text-gray-600 font-medium'>Gmail</h1>
            </div>

            {/* CENTER SECTION (SEARCH BAR) */}
            {user && (
                <div className="flex-1 flex justify-center">
                    <div className="flex items-center bg-[#EAF1FB] px-3 py-3 rounded-full w-[60%]">
                        <IoIosSearch size={22} className="text-gray-700" />
                        <input
                            type="text"
                            value={text}
                            onChange={(e) => setText(e.target.value)}
                            placeholder="Search Mail"
                            className="rounded-full w-full bg-transparent outline-none px-2"
                        />
                    </div>
                </div>
            )}

            {/* RIGHT SECTION */}
            {user && (
                <div className="flex items-center gap-4">

                    {/* Email Classifier Button */}
                    <button
                        onClick={() => window.open("http://127.0.0.1:5500/index.html", "_blank")}
                        className="px-4 py-2 bg-blue-500 text-white rounded-full hover:bg-blue-600 transition"
                    >
                        Email Classifier
                    </button>

                    <CiCircleQuestion size={24} className="cursor-pointer hover:text-gray-600" />
                    <IoIosSettings size={24} className="cursor-pointer hover:text-gray-600" />
                    <TbGridDots size={24} className="cursor-pointer hover:text-gray-600" />

                    <span
                        onClick={logoutHandler}
                        className='underline cursor-pointer hover:text-red-600'
                    >
                        Logout
                    </span>

                    <Avatar src={user.profilePhoto} size="40" round={true} />
                </div>
            )}
        </div>
    )
}

export default Navbar;
