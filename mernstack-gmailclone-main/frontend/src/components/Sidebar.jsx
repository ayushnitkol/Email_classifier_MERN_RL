import React from "react";
import { IoMdStar } from "react-icons/io";
import { LuPencil } from "react-icons/lu";
import { MdInbox, MdOutlineDrafts, MdOutlineKeyboardArrowDown, MdOutlineWatchLater } from "react-icons/md";
import { TbSend2 } from "react-icons/tb";
import { useDispatch } from "react-redux";
import { setOpen } from "../redux/appSlice";

const sidebarItems = [
  { icon: <MdInbox size={'18px'} />, text: "Inbox" },
  { icon: <IoMdStar size={'18px'} />, text: "Starred" },
  { icon: <MdOutlineWatchLater size={'18px'} />, text: "Snoozed" },
  { icon: <TbSend2 size={'18px'} />, text: "Sent" },
  { icon: <MdOutlineDrafts size={'18px'} />, text: "Drafts" },
  { icon: <MdOutlineKeyboardArrowDown size={'18px'} />, text: "More" },
];

const Sidebar = () => {
  const dispatch = useDispatch();
  return (
    <aside className="w-72 p-4">
      <div className="mb-4">
        <button
          onClick={() => dispatch(setOpen(true))}
          className="w-full flex items-center gap-3 justify-center bg-gradient-to-r from-blue-50 to-indigo-50 text-primary px-4 py-3 rounded-2xl hover:shadow-md"
        >
          <LuPencil size="20px" /> <span className="font-medium">Compose</span>
        </button>
      </div>

      <nav className="space-y-2">
        {sidebarItems.map((it, idx) => (
          <div key={idx} className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-gray-100 cursor-pointer">
            <div className="text-gray-600">{it.icon}</div>
            <div className="text-gray-800">{it.text}</div>
          </div>
        ))}
      </nav>
    </aside>
  );
};

export default Sidebar;
